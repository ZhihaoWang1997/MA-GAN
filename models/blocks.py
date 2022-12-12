
import torch
import torch.nn as nn
import torch.nn.functional as F

# 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. Default: 'nearest'


class Upx2(nn.Module):
    def __init__(self):
        super(Upx2, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class UpBlock(nn.Module):
    def __init__(self):
        super(UpBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
        )

    def forward(self, x):
        feas = F.interpolate(x,  scale_factor=2, mode='nearest')
        feas = self.body(feas)
        return feas


class PA(nn.Module):
    """PA is pixel attention"""
    def __init__(self, nf=64):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class AttentionLayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.con_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.con_du(y)
        return x * y


class PyConv2dAttention(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels for output feature map
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, r=2, pyconv_kernels=None, pyconv_groups=None,
                 stride=1, dilation=1, bias=False, attention=False):
        super(PyConv2dAttention, self).__init__()

        if pyconv_groups is None:
            pyconv_groups = [1, 4, 8]
        if pyconv_kernels is None:
            pyconv_kernels = [3, 5, 7]

        d = max(int(in_channels / r), 32)
        self.attention = attention

        split_channels = self._set_channels(out_channels, len(pyconv_kernels))
        self.pyconv_levels = nn.ModuleList([])
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels.append(nn.Conv2d(in_channels, split_channels[i], kernel_size=pyconv_kernels[i],
                                                stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                                dilation=dilation, bias=bias))

        # self.fc = nn.Linear(out_channels, d)
        # self.fcs = nn.ModuleList([])
        # for _ in range(len(pyconv_kernels)):
        #     self.fcs.append(nn.Linear(d, out_channels))
        #
        # self.softmax = nn.Softmax(dim=1)

        self.att1 = AttentionLayer(split_channels[0])
        self.att2 = AttentionLayer(split_channels[1])
        self.att3 = AttentionLayer(split_channels[2])

    def forward(self, x):
        feas = [layer(x) for layer in self.pyconv_levels]
        out = [self.att1(feas[0]), self.att2(feas[1]), self.att3(feas[2])]
        # out = [feas[0], feas[1], feas[2]]
        out = torch.cat(out, dim=1)
        return out

    def _set_channels(self, out_channels, levels):
        if levels == 1:
            split_channels = [out_channels]
        elif levels == 2:
            split_channels = [out_channels // 2 for _ in range(2)]
        elif levels == 3:
            split_channels = [out_channels // 4, out_channels // 4, out_channels // 2]
        elif levels == 4:
            split_channels = [out_channels // 4 for _ in range(4)]
        else:
            raise NotImplementedError
        return split_channels


class RDBN(nn.Module):
    def __init__(self, nf=64, gc=32, n=3):
        super(RDBN, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(n):
            if i == n - 1:
                # self.conv.add_module('Conv%d' % i, nn.Conv2d(nf + gc * i, nf, 3, padding=1))
                self.conv.add_module('PyConv%d' % i, PyConv2dAttention(nf + gc * i, nf, attention=True))
            else:
                self.conv.add_module('Conv%d' % i, nn.Conv2d(nf + gc * i, gc, 3, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.n = n

    def forward(self, x):
        out = [x]

        for i, layer in enumerate(self.conv):
            if i == 0:
                out.append(self.lrelu(layer(x)))
            elif i == self.n - 1:
                out.append(layer(torch.cat(out, dim=1)))
            else:
                out.append(self.lrelu(layer(torch.cat(out, dim=1))))

        y = out[-1]
        return x + y


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


if __name__ == "__main__":
    model = DeconvBlock(64, 64)
    a = torch.rand(1, 64, 64, 64)
    b = model(a)
    print(b.shape)
    print('end')
