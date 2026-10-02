# model/attention_unet2d.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate from Attention U-Net.

    g: decoder gating feature
    x: encoder skip feature
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(
                g,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet2D(nn.Module):
    """
    Classic 2D Attention U-Net for binary medical image segmentation.

    Output:
        logits: [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=(64, 128, 256, 512, 1024),
    ):
        super().__init__()

        f1, f2, f3, f4, f5 = features

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(in_channels, f1)
        self.conv2 = ConvBlock(f1, f2)
        self.conv3 = ConvBlock(f2, f3)
        self.conv4 = ConvBlock(f3, f4)
        self.conv5 = ConvBlock(f4, f5)

        self.up5 = UpConv(f5, f4)
        self.att5 = AttentionBlock(F_g=f4, F_l=f4, F_int=f3)
        self.up_conv5 = ConvBlock(f5, f4)

        self.up4 = UpConv(f4, f3)
        self.att4 = AttentionBlock(F_g=f3, F_l=f3, F_int=f2)
        self.up_conv4 = ConvBlock(f4, f3)

        self.up3 = UpConv(f3, f2)
        self.att3 = AttentionBlock(F_g=f2, F_l=f2, F_int=f1)
        self.up_conv3 = ConvBlock(f3, f2)

        self.up2 = UpConv(f2, f1)
        self.att2 = AttentionBlock(F_g=f1, F_l=f1, F_int=max(f1 // 2, 1))
        self.up_conv2 = ConvBlock(f2, f1)

        self.out_conv = nn.Conv2d(f1, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # Decoder + Attention Gates
        d5 = self.up5(x5)
        x4_att = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.up_conv2(d2)

        logits = self.out_conv(d2)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits


def initialize_attention_unet2d_binary_a(in_channels=3, out_channels=1):
    return AttentionUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
    )

def initialize_attention_unet2d_binary_b(in_channels=3, out_channels=1):
    return AttentionUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
    )