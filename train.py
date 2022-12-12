# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2021/3/16

import os
import time
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import json


from models.generator import Generator
from models.discriminator import Discriminator
from models.feature_loss import FeatureExtractor
from utils import calc_psnr, code_backup
from .dataset import get_dataset


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device(2)    
    sub_ = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(device)
    div_ = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(device)
    
    with open('config.json', mode='r') as f:
        cfg = json.load(f) 
    
    # ------------------------------------------------------------
    if not cfg["seed"]:
        seed = random.randint(0, 1000)
    else:
        seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------
    train_set = get_dataset(cfg["scale_factor"], cfg["train_set"])
    test_set = get_dataset(cfg["scale_factor"], cfg["test_set"] )
    training_data_loader = DataLoader(dataset=train_set, batch_size=cfg["train_batch_size"], shuffle=True, drop_last=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=cfg["test_batch_size"], shuffle=False)

    # ------------------------------------------------------------
    time_now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    root_path = f'results/fusion_nb_{cfg["nb"]}_{cfg["scale_factor"]}_{cfg["tag"]}_{time_now}'
    log_dir = f"{root_path}/logs"
    if os.path.exists(root_path):
        os.makedirs(root_path)
        os.makedirs(log_dir)
    code_backup(root_path)

    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------
    generator = Generator(cfg["nb"], cfg["scale_factor"]).to(device)
    discriminator = Discriminator().to(device)

    feature = FeatureExtractor().to(device)
    feature.eval()
    mse_loss = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss().to(device)
    bce_loss = torch.nn.BCELoss().to(device)
    sl1_loss = torch.nn.SmoothL1Loss().to(device)
    # ce_loss = torch.nn.CrossEntropyLoss().cuda()

    optimizer_G = optim.Adam(generator.parameters(), lr=cfg["lr_g"], weight_decay=0.0001, betas=(0.5, 0.999))
    schedule_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[25, 50, 75, 100], gamma=0.5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg["lr_d"], weight_decay=0.0001, betas=(0.5, 0.999))
    schedule_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[25, 50, 75, 100], gamma=0.5)

    global_best_psnr = 0.

    print(f"models save to {root_path}")
    for epoch in range(1, cfg["epochs"] + 1):
        print(f'--------Epoch {epoch} Start Training x {cfg["scale_factor"]}---------')
        generator.eval()
        discriminator.train()
        g_loss = list()
        d_loss = list()
        for batch_id, (lr, hr, _) in enumerate(tqdm(training_data_loader)):
            n = batch_id + (epoch - 1) * len(training_data_loader)
            lr = lr.to(device)
            hr = hr.to(device)

            lr = (lr - sub_) / div_
            hr = (hr - sub_) / div_

            one_label = torch.ones(size=[cfg["train_batch_size"]], dtype=torch.float32, device=device)
            zero_label = torch.zeros(size=[cfg["train_batch_size"]], dtype=torch.float32, device=device)

            optimizer_D.zero_grad()

            d_out_real = discriminator(hr)
            err_d_real = bce_loss(d_out_real, one_label)
            
            gen_hr = generator(lr)
            d_out_fake = discriminator(gen_hr)
            err_d_fake = bce_loss(d_out_fake, zero_label)
            
            err_d_total = err_d_real + err_d_fake
            err_d_total.backward()

            optimizer_D.step()
            
            optimizer_G.zero_grad()
            gen_hr = generator(lr)

            d_out_fake_1 = discriminator(gen_hr)
            err_g = bce_loss(d_out_fake_1, one_label)

            err_feas = l1_loss(feature(gen_hr), feature(hr))
            err_pixel = sl1_loss(gen_hr, hr)

            err_g_total = 1e-3 * err_g + 1e-2 * err_feas + err_pixel
            err_g_total.backward()
            optimizer_G.step()
            
            g_loss.append(err_g_total.item())
            d_loss.append(err_d_total.item())
            writer.add_scalar('Train/err_g', err_g.item(), n)
            writer.add_scalar('Train/err_feas', err_feas.item(), n)
            writer.add_scalar('Train/err_pixel', err_pixel.item(), n)
            writer.add_scalar('Train/err_g_total', err_g_total.item(), n)
            writer.add_scalar('Train/err_d_fake', err_d_fake.item(), n)
            writer.add_scalar('Train/err_d_real', err_d_real.item(), n)
            writer.add_scalar('Train/err_d_total', err_d_total.item(), n)

            # break

        g_loss_mean = np.array(g_loss).mean()
        d_loss_mean = np.array(d_loss).mean()
        print(f"Epoch:{epoch}\tloss D:{d_loss_mean}\tloss G:{g_loss_mean}")
        print('--------Start Testing---------')
        with torch.no_grad():
            generator.eval()
            discriminator.eval()

            psnr_list = list()
            for batch_id, (lr, hr, _) in enumerate(tqdm(testing_data_loader)):
                n = batch_id + (epoch - 1) * len(testing_data_loader)
                lr = lr.to(device)
                hr = hr.to(device)

                lr = (lr - sub_) / div_
                gen_hr = generator(lr)

                gen_hr = gen_hr * div_ + sub_
                gen_hr.clamp_(0, 1)

                mse = mse_loss(gen_hr, hr)
                psnr = calc_psnr(gen_hr, hr)
                psnr_list.append(psnr)
                writer.add_scalar("Test/PSNR", psnr, n)
                assert gen_hr.size() == hr.size()

            new_psnr = np.array(psnr_list).mean()
            if global_best_psnr < new_psnr:
                global_best_psnr = new_psnr
                state = {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "lr_g": cfg["lr_g"],
                    "lr_d": cfg["lr_d"],
                    "seed": seed,
                    "train_batch_size": cfg["train_batch_size"],
                    "best psnr": global_best_psnr
                }
                torch.save(state, f"{root_path}/best_model.pth")

                with open(f"{root_path}/best_psnr.txt", mode='w', encoding='utf-8') as f:
                    f.write(str(global_best_psnr))
            print(f"Testing psnr mean:{new_psnr} | Global best psnr:{global_best_psnr}")

        schedule_D.step()
        schedule_G.step()

    print(f"finish training, best psnr:{global_best_psnr}")
    print(f"models save to {root_path}")

    exit(0)












