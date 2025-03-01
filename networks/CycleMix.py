import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CycleMix2D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, in_channels=1, is_batchnorm=True):
        """
        Initializes the CycleMix model with a backbone 2D U-Net segmentation network.
        """
        super(CycleMix2D, self).__init__()
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling
        self.conv1 = UnetConv2D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2D(filters[3], filters[4], self.is_batchnorm)

        # Upsampling
        self.up_concat4 = UnetUp2D(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp2D(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp2D(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp2D(filters[1], filters[0], is_batchnorm)

        # Final convolution
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def mix_augmentation(self, x1, x2, alpha=0.5, apply_occlusion=True):
        """
        Perform mixup augmentation with optional random occlusion.
        """
        mixed_x = alpha * x1 + (1 - alpha) * x2

        if apply_occlusion:
            mask = torch.ones_like(mixed_x)
            mask[:, :, :32, :32] = 0  # Example occlusion area
            mixed_x = mixed_x * mask

        return mixed_x

    def forward(self, x1, x2):
        """
        Forward pass with CycleMix's mix augmentation and regularization.
        """
        # Standard pass for both inputs
        pred1 = self._forward_single(x1)
        pred2 = self._forward_single(x2)

        # Mixed augmentation
        mixed_x = self.mix_augmentation(x1, x2)
        mixed_pred = self._forward_single(mixed_x)

        return pred1, mixed_pred

    def _forward_single(self, x):
        """
        Forward pass through the 2D U-Net backbone.
        """
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        return self.final(up1)


class UnetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(UnetConv2D, self).__init__()
        self.is_batchnorm = is_batchnorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if is_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.is_batchnorm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
        return x


class UnetUp2D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetUp2D, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv = UnetConv2D(out_size + out_size, out_size, is_batchnorm)

    def forward(self, high_feature, low_feature):
        upsampled = self.up(low_feature)
        concatenated = torch.cat([upsampled, high_feature], dim=1)
        return self.conv(concatenated)


def cycle_mix_loss(pred1, pred2, mixed_pred, target1, target2, alpha=0.5):
    """
    Compute the CycleMix loss combining supervision and regularization.
    """
    criterion = nn.CrossEntropyLoss()

    loss1 = criterion(pred1, target1)
    loss2 = criterion(pred2, target2)

    mixed_target = alpha * target1 + (1 - alpha) * target2
    mixed_loss = criterion(mixed_pred, mixed_target.long())

    return loss1 + loss2 + mixed_loss


# Example Usage
if __name__ == "__main__":
    batch_size = 2
    channels = 3
    height, width = 256, 256
    n_classes = 6

    # Example data
    x1 = torch.randn(batch_size, channels, height, width)  # Input 1
    x2 = torch.randn(batch_size, channels, height, width)  # Input 2
    # y1 = torch.randint(0, n_classes, (batch_size, height, width))  # Label 1
    # y2 = torch.randint(0, n_classes, (batch_size, height, width))  # Label 2

    # Initialize the model
    model = CycleMix2D(feature_scale=4, n_classes=n_classes, in_channels=channels, is_batchnorm=True)

    # Forward pass
    pred1, pred2 = model(x1, x2)

    # Compute loss
    # loss = cycle_mix_loss(pred1, pred2, mixed_pred, y1, y2)
    # print(f"Loss: {loss.item():.4f}")
