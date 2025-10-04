from torchinfo import summary
import torch.nn as nn
import torch


class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        f = nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
        return x * f

class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        f = nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
        return f

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = hswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channel, out_channel // reduction, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(out_channel // reduction, out_channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = out * self.se(out)
        out += self.shortcut(x)
        out = nn.functional.relu(out, inplace=True)
        return out

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV3Small, self).__init__()

        self.conv1 = ConvBlock(1, 16, 3, 2, 1)     # 1/2
        self.bottlenecks = nn.Sequential(
            ResidualBlock(16, 16, 3, 2, False),     # 1/4
            ResidualBlock(16, 32, 3, 2, False),     # 1/8
            ResidualBlock(32, 64, 3, 1, False),
            ResidualBlock(64, 64, 3, 1, True),
        )
        self.conv2 = ConvBlock(64, 96, 1, 1, 0, groups=2)
        self.conv3 = nn.Conv1d(96, 120, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(120)
        self.act = hswish()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(120, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bottlenecks(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

