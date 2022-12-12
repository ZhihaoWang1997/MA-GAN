import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.blocks import *


class Generator(nn.Module):
    def __init__(self, nb, scale):
        super(Generator, self).__init__()

        assert scale in [2, 4, 8]
        self.scale = scale
        self.n = int(math.log2(scale))

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )

        self.body = []
        for _ in range(nb):
            self.body.append(RDBN())
        self.body = nn.Sequential(*self.body)

        self.uplayers = []
        for i in range(self.n):
            # upblock
            self.uplayers.append(UpBlock())
            self.uplayers.append(PA())

            # # shuffle layer
            # self.uplayers.append(nn.Conv2d(64, 64 * 4, kernel_size=3, stride=1, padding=1))
            # self.uplayers.append(nn.PixelShuffle(2))

            # deconv
            # self.uplayers.append(DeconvBlock(64, 64))

        self.uplayers = nn.Sequential(*self.uplayers)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.act_last = nn.Tanh()

        self.branch = nn.Sequential(nn.Conv2d(6, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.05, inplace=True),
                                    nn.AdaptiveAvgPool2d(1))

        self.fc1 = nn.Linear(64, 3)
        self.fc2 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        # noise = torch.rand(size=images.size()).cuda()
        image_inter = F.interpolate(images, scale_factor=self.scale, mode='bilinear', align_corners=False)

        x = self.head(images)
        x = self.body(x)
        x = self.uplayers(x)
        x = self.conv_last(x)
        x = self.act_last(x)
        # return x + image_inter
        # return (x+1)/2

        fusion = torch.cat([x, image_inter], dim=1)
        fusion_1 = torch.cat([x.unsqueeze(1), image_inter.unsqueeze(1)], dim=1)
        fea_z = self.branch(fusion)
        fea_z.squeeze_(-1).squeeze_(-1)
        
        v1 = self.fc1(fea_z).unsqueeze(1)
        v2 = self.fc2(fea_z).unsqueeze(1)
        v = torch.cat([v1, v2], dim=1)
        v = self.softmax(v)
        v = v.unsqueeze(-1).unsqueeze(-1)
        return (fusion_1 * v).sum(dim=1)




if __name__ == '__main__':
    net = Generator(4, 4)
    out = net(torch.rand(4, 3, 16, 16))
    print(out.size())
