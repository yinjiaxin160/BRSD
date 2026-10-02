# model/unet2d.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0, use_instance_norm=True):
        super().__init__()
        norm = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels1,
            in_channels2,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 防止奇偶尺寸导致 concat 报错
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = torch.nn.functional.interpolate(
                x1,
                size=x2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Standard 2D U-Net for binary medical image segmentation.
    Uses InstanceNorm2d (not BatchNorm2d) for training stability with small batches.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128, 256),
        dropout=(0.05, 0.1, 0.2, 0.3, 0.5),
    ):
        super().__init__()

        self.in_conv = ConvBlock(in_channels, features[0], dropout[0])

        self.down1 = DownBlock(features[0], features[1], dropout[1])
        self.down2 = DownBlock(features[1], features[2], dropout[2])
        self.down3 = DownBlock(features[2], features[3], dropout[3])
        self.down4 = DownBlock(features[3], features[4], dropout[4])

        self.up1 = UpBlock(features[4], features[3], features[3], 0.0)
        self.up2 = UpBlock(features[3], features[2], features[2], 0.0)
        self.up3 = UpBlock(features[2], features[1], features[1], 0.0)
        self.up4 = UpBlock(features[1], features[0], features[0], 0.0)

        self.out_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        return self.out_conv(x)


def initialize_unet2d_binary_a(in_channels=3, out_channels=1,  features=(64, 128, 256, 512, 512)):
    """
    Binary-seg dataset A:
        input: RGB image
        output: one-channel foreground logits
    Args:
        features: channel list for encoder/decoder, default (16, 32, 64, 128, 256)
    """
    return UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
    )

# Binary-seg dataset B:
def initialize_unet2d_binary_b(in_channels=3, out_channels=1, features=(64, 128, 256, 512, 512)):
    """
    Binary-seg dataset B:
        input: RGB image
        output: one-channel foreground logits
    Args:
        features: channel list for encoder/decoder, default (64, 128, 256, 512, 512)
    """
    return UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
    )

# BraTS2020 binary segmentation:
def initialize_unet2d_brats2020(in_channels=4, out_channels=4, features=(16, 32, 64, 128, 256)):
    """
    BraTS2020 binary segmentation:
        input: RGB image
        output: one-channel foreground logits
    Args:
        features: channel list for encoder/decoder, default (16, 32, 64, 128, 256)
    """
    return UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
    )
        