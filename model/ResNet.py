from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 简化的残差函数 - 减少通道数和移除激活函数
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),  # 使用bias增加过拟合
            # 移除BatchNorm以降低性能
            # 移除ReLU激活函数
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=True),
            # 移除BatchNorm
        )

        # 简化的shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=True),
                # 移除BatchNorm
            )

    def forward(self, x):
        # 移除激活函数以降低表达能力
        return self.residual_function(x) + self.shortcut(x)


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=9, input_channels=6):
        super(RSNet, self).__init__()

        self.in_channels = 16  # 减少初始通道数

        # 简化的初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1, stride=1, bias=True),  # 使用bias
            # 移除BatchNorm和ReLU
        )

        # 减少层数
        self.layer1 = self._make_layer(block, 32, num_block[0], 2)  # 减少通道数
        # 移除layer2
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 简化的分类器
        self.fc = nn.Linear(32 * block.expansion, num_classes)  # 减少输入特征

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, x  # 返回两个相同的输出以保持接口一致


def ResNet_6(num_classes=9, input_channels=6, **kwargs):
    return RSNet(BasicBlock, [1], num_classes=num_classes, input_channels=input_channels)  # 减少块数

