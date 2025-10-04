import torch
import torch.nn as nn
from torchinfo import summary


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # 多层时添加dropout
        )

        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

        # 可选：添加BatchNorm和Dropout
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 可选：应用BatchNorm和Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # 全连接层
        out = self.fc(out)

        # 返回主输出和辅助输出（相同）
        return out, out


def LSTM(input_size=6, hidden_size=128, num_layers=2, num_classes=9, **kwargs):
    """创建LSTM模型

    参数:
        input_size (int): 每个时间步的特征数量 (默认 6)
        hidden_size (int): LSTM隐藏层大小 (默认 128)
        num_layers (int): LSTM层数 (默认 2)
        num_classes (int): 输出类别数 (默认 9)
    """
    return LSTMClassifier(input_size, hidden_size, num_layers, num_classes)