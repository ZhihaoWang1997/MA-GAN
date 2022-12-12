# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2021/3/18

import os
from shutil import copyfile, copytree
import glob

import torch
import skimage.metrics as metrics


def calc_psnr(x, y):
    x = x.cpu().squeeze(0).numpy()
    y = y.cpu().squeeze(0).numpy()
    # psnr = metrics.peak_signal_noise_ratio(x, y)
    psnr = metrics.peak_signal_noise_ratio(x, y, data_range=1)
    return psnr


def code_backup(save_path):
    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_path, "codes")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for py_file in glob.glob(os.path.join(current_path, "*.py")):
        # copyfile(py_file, save_path)
        copyfile(py_file, os.path.join(save_path, py_file.split("/")[-1]))
    copytree(os.path.join(current_path, "dataset"), os.path.join(save_path, "dataset"))
    copytree(os.path.join(current_path, "models"), os.path.join(save_path, "models"))


if __name__ == "__main__":
    # a = torch.rand(1, 3, 16, 16).cuda()
    # b = torch.rand(1, 3, 16, 16).cuda()
    # psnr = calc_psnr(a, b)
    # print(psnr)

    with open('best psnr.txt', mode='w', encoding='utf-8') as f:
        f.write("654564646")

