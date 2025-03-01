import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct
import torch.nn.functional as F
import torch.nn as nn
import argparse
from skimage import io, transform
from .MedSAM_Inference import *

class UNet2D(nn.Module):
    """
    A simple 2D U-Net architecture for coarse segmentation.
    """

    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.center = self.conv_block(512, 1024)

        self.decoder4 = self.up_conv(1536, 512)
        self.decoder3 = self.up_conv(768, 256)
        self.decoder2 = self.up_conv(384, 128)
        self.decoder1 = self.up_conv(192, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(in_channels, out_channels),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        center = self.center(F.max_pool2d(e4, 2))

        d4 = self.decoder4(torch.cat([F.interpolate(center, scale_factor=2), e4], dim=1))
        d3 = self.decoder3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        return self.final(d1)

class FrequencyTransformBlock(nn.Module):
    """Frequency transform block with Channel and Spatial Frequency Transform."""

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Frequency Transform (CFT)
        batch_size, channels, height, width = x.size()
        channel_frequency = torch.fft.fft2(x, dim=(-2, -1)).real.mean(dim=(-2, -1))  # Simulating DCT for each channel
        channel_frequency = self.sigmoid(self.fc(channel_frequency))
        x_cft = x * channel_frequency.view(batch_size, channels, 1, 1)

        # Spatial Frequency Transform (SFT)
        spatial_frequency = torch.fft.fft2(x.mean(dim=1, keepdim=True), dim=(-2, -1)).real.mean(dim=1)
        spatial_frequency = spatial_frequency.unsqueeze(1).repeat(1, channels, 1, 1)
        x_sft = x * spatial_frequency

        # Combining both transformations
        return x_cft + x_sft

class MultiScaleFrequencyTransform(nn.Module):
    """Applies multi-scale atrous convolutions and frequency transforms."""

    def __init__(self, in_channels, out_channels, rates=(1, 2, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
            for rate in rates
        ])
        self.frequency_transform = FrequencyTransformBlock(out_channels)

    def forward(self, x):
        # Multi-scale Feature Map
        x_msfm = sum(conv(x) for conv in self.convs)
        # Frequency Transform
        x_ft = x_msfm + self.frequency_transform(x_msfm)
        return x_ft

class CSFNet(nn.Module):
    """Channel-Spatial Frequency Network (CSFNet)"""

    def __init__(self, in_channels=1, num_classes=4, base_channels=64):
        super().__init__()
        # Encoder Stages
        self.en_stage1 = MultiScaleFrequencyTransform(in_channels, base_channels)
        self.down1 = nn.MaxPool2d(2)

        self.en_stage2 = MultiScaleFrequencyTransform(base_channels, base_channels * 2)
        self.down2 = nn.MaxPool2d(2)

        self.en_stage3 = MultiScaleFrequencyTransform(base_channels * 2, base_channels * 4)
        self.down3 = nn.MaxPool2d(2)

        self.en_stage4 = MultiScaleFrequencyTransform(base_channels * 4, base_channels * 8)
        self.down4 = nn.MaxPool2d(2)

        self.en_stage5 = MultiScaleFrequencyTransform(base_channels * 8, base_channels * 16)

        # Decoder Stages
        self.de_stage4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.de_stage3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.de_stage2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.de_stage1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)

        # Output Segmentation Head
        self.segmentation_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.en_stage1(x)
        x2 = self.en_stage2(self.down1(x1))
        x3 = self.en_stage3(self.down2(x2))
        x4 = self.en_stage4(self.down3(x3))
        x5 = self.en_stage5(self.down4(x4))

        # Decoder with skip connections
        d4 = self.de_stage4(x5) + x4
        d3 = self.de_stage3(d4) + x3
        d2 = self.de_stage2(d3) + x2
        d1 = self.de_stage1(d2) + x1

        # Segmentation Head
        out = self.segmentation_head(d1)
        return out


class ProgressMix(nn.Module):

    def __init__(self, in_chns, class_num):
        super(ProgressMix, self).__init__()
        self.CSFNet = CSFNet(in_channels=in_chns, num_classes=class_num)

    def forward(self, x):
        output1 = self.CSFNet(x)

        # _, _, H, W = x.shape
        # #  image preprocessing
        # img_1024 = transform.resize(
        #     x, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        # ).astype(np.uint8)
        # img_1024 = (img_1024 - img_1024.min()) / np.clip(
        #     img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        # )
        # # convert the shape to (3, H, W)
        # img_1024_tensor = (
        #     torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        # )

        # box_np = np.array([[int(output1) for output1 in args.box[1:-1].split(',')]])
        # # transfer box_np t0 1024x1024 scale
        # box_prompts = box_np / np.array([W, H, W, H]) * 1024

        # from MedSAM.segment_anything import sam_model_registry
        # with torch.no_grad():
        #     image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
        # output2 = medsam_inference(medsam_model, image_embedding, box_prompts, H, W)

        # return output1, output2
        return output1


# Example Usage
if __name__ == "__main__":
    network1 = ProgressMix()
    x = torch.randn(1, 1, 256, 256)  # Input image
    output1, output2 = network1(x)
    print(output1.shape)
    print(output2.shape)

