import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelSSM(nn.Module):
    """
    Pixel-level Structured State Space Model (SSM).
    Operates on local sub-kernels (sub-windows) of the feature map.
    """

    def __init__(self, in_channels, kernel_size):
        super(PixelSSM, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class PatchSSM(nn.Module):
    """
    Patch-level Structured State Space Model (SSM).
    Operates on aggregated patch-level features.
    """

    def __init__(self, in_channels):
        super(PatchSSM, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = self.pool(x).view(b, c)
        out = self.fc(pooled)
        out = out.view(b, c, 1, 1).expand_as(x)
        return x + out


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba block for local-global dependency modeling.
    Combines PixelSSM and PatchSSM.
    """

    def __init__(self, in_channels, kernel_size):
        super(BidirectionalMamba, self).__init__()
        self.pixel_ssm = PixelSSM(in_channels, kernel_size)
        self.patch_ssm = PatchSSM(in_channels)

    def forward(self, x):
        x_pixel = self.pixel_ssm(x)
        x_patch = self.patch_ssm(x_pixel)
        return x_patch


class LMBlock(nn.Module):
    """
    Large Kernel Mamba Block combining pixel and patch SSM.
    """

    def __init__(self, in_channels, kernel_size):
        super(LMBlock, self).__init__()
        self.mamba = BidirectionalMamba(in_channels, kernel_size)

    def forward(self, x):
        return self.mamba(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class LKMUNet(nn.Module):
    """
    Large Kernel Mamba U-Net for medical image segmentation.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(LKMUNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)


        # self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        # self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        # self.encoder4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(2)

        self.lm_block1 = LMBlock(64, kernel_sizes[0])
        self.lm_block2 = LMBlock(128, kernel_sizes[1])
        self.lm_block3 = LMBlock(256, kernel_sizes[2])
        self.lm_block4 = LMBlock(512, kernel_sizes[3])

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.decoder3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.lm_block1(self.encoder1(x))
        e2 = self.lm_block2(self.encoder2(self.pool(e1)))
        e3 = self.lm_block3(self.encoder3(self.pool(e2)))
        e4 = self.lm_block4(self.encoder4(self.pool(e3)))

        # Decoder
        d3 = self.decoder3(torch.cat([self.upconv3(e4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.final_conv(d1)