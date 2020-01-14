import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        bias=False
    )


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(ResidualBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, norm_layer=None):
        super(ResNet, self).__init__()

        self.in_channels = 64
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


tt = nn.Conv2d(1, 1, kernel_size=7, stride=2, padding=3, bias=False)
out = tt(torch.ones(1, 1, 64, 64))
print(out.shape)