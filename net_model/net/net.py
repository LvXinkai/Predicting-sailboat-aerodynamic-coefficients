import math
from typing import Dict
import torch
import torch.nn as nn

channel_group_num = 8


class Conv_GN_ReLu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Conv_GN_ReLu, self).__init__()
        self.conv_gn_relu = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.GroupNorm(channel_group_num, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_gn_relu(x)
        return x


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_Attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Channel_Attention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_ch, in_ch // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_ch // ratio, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        return self.sigmoid(out)


class CSAB(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=16, kernel_size=3):
        super(CSAB, self).__init__()
        self.ca = Channel_Attention(in_ch, ratio)
        self.conv = Conv_GN_ReLu(in_ch, out_ch)
        self.sa = Spatial_Attention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = self.conv(out)
        result = self.sa(out)

        return result


class RSB(nn.Module): #  Residual-Subsampled Block
    def __init__(self, ch):
        super(RSB, self).__init__()
        self.channels = int(ch // 2)
        self.down_1 = nn.Conv3d(self.channels, self.channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.gn_1 = nn.GroupNorm(channel_group_num, self.channels)
        self.down_2 = nn.Conv3d(self.channels, self.channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.gn_2 = nn.GroupNorm(channel_group_num, self.channels)

        self.conv_1x1x1 = nn.Conv3d(ch, ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn_3 = nn.GroupNorm(channel_group_num, ch)

        self.relu = nn.ReLU(inplace=True)

        self.down_sample = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=1, stride=2, bias=False),
            nn.GroupNorm(channel_group_num, ch),
        )

    def forward(self, x):
        residual = self.down_sample(x)

        spx = torch.split(x, int(x.shape[1] // 2), 1)
        do1 = self.down_1(spx[0])
        do1 = self.relu(self.gn_1(do1))

        do2 = self.down_2(spx[1])
        do2 = self.relu(self.gn_2(do2))

        out = torch.cat((do1, do2), 1)
        out = self.conv_1x1x1(out)
        out = self.gn_3(out)

        out = out + residual
        out = self.relu(out)

        return out


class MRA(nn.Module):
    def __init__(self, inplanes, planes, stride=1, baseWidth=26, scale=4):
        super(MRA, self).__init__()
        width = int(math.floor(planes * (baseWidth / 32.0)))
        self.conv1 = nn.Conv3d(inplanes, width * scale, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(13, width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        gns = []
        for i in range(self.nums):
            convs.append(nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            gns.append(nn.GroupNorm(13, width))
        self.convs = nn.ModuleList(convs)
        self.gns = nn.ModuleList(gns)

        self.conv2 = nn.Conv3d(width * scale, planes, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(channel_group_num, planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(channel_group_num, planes),
            )
        self.scale = scale
        self.width = width
        self.csab = CSAB(in_ch=inplanes, out_ch=planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.gns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv2(out)
        out = self.gn2(out)

        residual = self.downsample(x)

        out = out * self.csab(x)

        out = out + residual
        out = self.relu(out)

        return out


class Encoder_3d(nn.Module):
    def __init__(self, inplanes, planes):
        super(Encoder_3d, self).__init__()
        self.mra = MRA(inplanes, planes)
        self.down = RSB(planes)

    def forward(self, x):
        x = self.mra(x)
        x = self.down(x)
        return x


class Up_Conv(nn.Module):
    def __init__(self, ch_in):
        super(Up_Conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_in, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Conv3d(ch_in, ch_in, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(channel_group_num, ch_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class RIB(nn.Module):  # Residual-Inception Blocks
    def __init__(self, ch_in, ch_out):
        super(RIB, self).__init__()
        ch_2 = int(ch_in // 2)

        self.a_conv_5x5x5 = nn.Conv3d(ch_2, ch_2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.gn_5x5x5 = nn.GroupNorm(channel_group_num, ch_2)

        self.a_conv_7x7x7 = nn.Conv3d(ch_2, ch_2, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.gn_7x7x7 = nn.GroupNorm(channel_group_num, ch_2)

        self.conv_1x1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, bias=False)
        self.gn_1x1x1 = nn.GroupNorm(channel_group_num, ch_out)

        self.relu = nn.ReLU(inplace=True)

        self.residual = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(channel_group_num, ch_out),
        )

    def forward(self, x):
        residual = self.residual(x)

        spx = torch.split(x, int(x.shape[1] // 2), 1)
        con_5 = self.a_conv_5x5x5(spx[0])
        con_5 = self.relu(self.gn_5x5x5(con_5))
        con_5 = self.relu(con_5 + spx[0])

        con_7 = self.a_conv_7x7x7(spx[1])
        con_7 = self.relu(self.gn_7x7x7(con_7))
        con_7 = self.relu(con_7 + spx[1])

        out = torch.cat((con_5, con_7), 1)
        out = self.conv_1x1x1(out)
        out = self.gn_1x1x1(out)

        out = out + residual
        out = self.relu(out)

        return out


class Decoder_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Decoder_3d, self).__init__()

        self.up = Up_Conv(ch_in)
        self.rib = RIB(ch_in, ch_out)
        self.mra = MRA(ch_in, ch_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = self.rib(x1)
        out = self.mra(torch.cat([x2, out], dim=1))
        return out


class OutConv_3d(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv_3d, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1, bias=False)
        )


class MultiScale_ResUnet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 base_c: int = 32):

        super(MultiScale_ResUnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(in_channels, base_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(channel_group_num, base_c)
        self.relu = nn.ReLU(inplace=True)

        self.en_layer1 = Encoder_3d(base_c, base_c * 2)
        self.en_layer2 = Encoder_3d(base_c * 2, base_c * 4)
        self.en_layer3 = Encoder_3d(base_c * 4, base_c * 8)
        self.en_layer4 = Encoder_3d(base_c * 8, base_c * 16)

        self.de_layer1 = Decoder_3d(base_c * 16, base_c * 8)
        self.de_layer2 = Decoder_3d(base_c * 8, base_c * 4)
        self.de_layer3 = Decoder_3d(base_c * 4, base_c * 2)
        self.de_layer4 = Decoder_3d(base_c * 2, base_c)
        self.out_conv = OutConv_3d(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x1 = self.relu(self.gn1(x))
        x2 = self.en_layer1(x1)
        x3 = self.en_layer2(x2)
        x4 = self.en_layer3(x3)
        x5 = self.en_layer4(x4)

        x = self.de_layer1(x5, x4)
        x = self.de_layer2(x, x3)
        x = self.de_layer3(x, x2)
        x = self.de_layer4(x, x1)
        out = self.out_conv(x)

        return out, x5
