import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import seaborn as sns
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# 忽略警告
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 标签映射
CARAGEENAN_MAPPING = {
    0.0: 0, 0.03: 1, 0.05: 2, 0.08: 3,
    0.10: 4, 0.12: 5, 0.15: 6, 0.17: 7, 0.20: 8
}

# 频率点
FREQUENCIES = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]

# 特征列表
FEATURES = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']


# LSTM 模型定义
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        # 初始化隐藏状态
        self.h0 = None
        self.c0 = None

    def forward(self, x):
        # 如果隐藏状态未初始化或批次大小不匹配，则重新初始化
        if self.h0 is None or self.h0.size(1) != x.size(0):
            self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (self.h0, self.c0))

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out, None


# 自定义数据集类
class CarageenanDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


# 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    # 读取Excel数据
    df = pd.read_excel(file_path)
    print(f"数据加载成功，共 {len(df)} 行")
    print(f"数据列名: {df.columns.tolist()}")

    # 检查必要的列是否存在
    required_columns = ['卡拉胶含量'] + [f"{feat}_{freq}" for feat in FEATURES for freq in FREQUENCIES]
    missing_cols = [col for col in required_columns if col not in df.columns]

    # 特殊处理Φ列（有时可能写作Phi）
    for freq in FREQUENCIES:
        phi_col = f"Φ_{freq}"
        if phi_col not in df.columns and f"Phi_{freq}" in df.columns:
            df.rename(columns={f"Phi_{freq}": phi_col}, inplace=True)
            print(f"已将列 'Phi_{freq}' 重命名为 'Φ_{freq}'")

    # 再次检查缺失列
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺失以下列: {missing_cols}")
        # 创建缺失列并填充0
        for col in missing_cols:
            df[col] = 0.0
        print("已用0填充缺失列")

    # 提取标签
    if '卡拉胶含量' in df.columns:
        labels = df['卡拉胶含量'].map(CARAGEENAN_MAPPING).values
        df = df.drop(columns=['卡拉胶含量'])
    else:
        raise ValueError("'卡拉胶含量' 列未找到")

    n_samples = len(df)
    n_freqs = len(FREQUENCIES)
    n_features = len(FEATURES)

    # 创建数据数组
    data = np.zeros((n_samples, n_freqs, n_features))

    # 填充数据
    for sample_idx in range(n_samples):
        for freq_idx, freq in enumerate(FREQUENCIES):
            for feat_idx, feat in enumerate(FEATURES):
                col_name = f"{feat}_{freq}"
                data[sample_idx, freq_idx, feat_idx] = df.loc[sample_idx, col_name]

    # 特征标准化
    scalers = {}
    for feat_idx in range(n_features):
        feat_data = data[:, :, feat_idx].reshape(-1, 1)
        scaler = StandardScaler()
        feat_data_scaled = scaler.fit_transform(feat_data)
        data[:, :, feat_idx] = feat_data_scaled.reshape(n_samples, n_freqs)
        scalers[feat_idx] = scaler

    print(f"数据预处理完成，形状: {data.shape}")
    return data, labels, scalers


# 训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output, _ = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算预测
        _, predicted = torch.max(output.data, 1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # 每10个batch打印一次
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算训练指标
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f'\n训练集: 平均损失: {avg_loss:.4f}, 准确率: {acc:.4f}, 精确率: {precision:.4f}, '
          f'召回率: {recall:.4f}, F1分数: {f1:.4f}')

    return avg_loss, acc


# 测试函数
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 前向传播
            output, _ = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

            # 计算预测
            _, predicted = torch.max(output.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算测试指标
    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f'\n测试集: 平均损失: {avg_loss:.4f}, 准确率: {acc:.4f}, 精确率: {precision:.4f}, '
          f'召回率: {recall:.4f}, F1分数: {f1:.4f}')

    # 打印混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    print("\n混淆矩阵:")
    print(cm)

    # 打印分类报告
    reverse_mapping = {v: f"{k * 100:.0f}%" for k, v in CARAGEENAN_MAPPING.items()}
    target_names = [reverse_mapping[i] for i in range(9)]

    print("\n分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=target_names))

    return avg_loss, acc, f1


# 绘制训练历史
def plot_training_history(train_losses, test_losses, train_accs, test_accs, save_dir):
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('训练和测试损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(test_accs, label='测试准确率')
    plt.title('训练和测试准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path)
    print(f"训练历史图已保存到: {save_path}")
    plt.close()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, save_dir, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = '归一化' + title
        print("归一化混淆矩阵")
    else:
        print('未归一化混淆矩阵')

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout()

    # 保存混淆矩阵
    suffix = "_normalized" if normalize else ""
    save_path = os.path.join(save_dir, f'confusion_matrix{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {save_path}")
    plt.close()

    return save_path


# 主函数
if __name__ == '__main__':
    # 配置参数
    config = {
        'data_path': 'D:/谱/BeefClassifier/data/zq_spectra_data.xlsx',  # 数据文件路径
        'test_size': 0.2,  # 测试集比例
        'random_seed': 42,  # 随机种子
        'batch_size': 32,  # 批大小
        'learning_rate': 0.01,  # 学习率
        'num_epochs': 50,  # 训练轮数
        'input_size': 6,  # 输入特征数
        'hidden_size': 128,  # LSTM隐藏层大小
        'num_layers': 2,  # LSTM层数
        'num_classes': 9,  # 类别数
        'save_dir': 'checkpoints/LSTM-h',  # 模型保存目录
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"所有结果将保存在: {config['save_dir']}")

    # 加载并预处理数据
    print("=" * 50)
    print("加载和预处理数据...")
    start_time = time.time()

    try:
        data, labels, scalers = load_and_preprocess_data(config['data_path'])
        print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")
        print(f"数据加载和预处理耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"数据加载错误: {e}")
        raise

    # 创建数据集
    dataset = CarageenanDataset(data, labels)

    # 划分训练集和测试集
    dataset_size = len(dataset)
    test_size = int(config['test_size'] * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"数据加载完成: {len(train_dataset)} 训练样本, {len(test_dataset)} 测试样本")
    print("=" * 50)

    # 初始化模型
    model = LSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    ).to(device)

    # 打印模型结构
    print("\n模型架构:")
    print(model)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params / 1e6:.2f}M, 可训练参数: {trainable_params / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.001)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.95, patience=3, verbose=True
    )

    # 训练循环
    best_f1 = 0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    print("\n开始训练...")
    print("=" * 50)
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n轮次 {epoch}/{config['num_epochs']}")

        # 训练
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)

        # 测试
        test_loss, test_acc, test_f1 = test(model, test_loader, criterion)

        # 记录指标
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 更新学习率
        scheduler.step(test_f1)

        # 保存最佳模型
        if test_f1 > best_f1:
            best_f1 = test_f1
            model_path = os.path.join(config['save_dir'], f'best_lstm-h_epoch{epoch}_f1_{test_f1:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_f1': test_f1,
                'config': config,
            }, model_path)
            print(f"保存最佳模型到 {model_path}, F1分数: {test_f1:.4f}")

    # 训练结束
    training_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"训练完成，耗时: {training_time // 60:.0f}分 {training_time % 60:.0f}秒")
    print(f"最佳验证F1分数: {best_f1:.4f}")

    # 保存最终模型
    final_model_path = os.path.join(config['save_dir'], 'final_lstm-h_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"保存最终模型到 {final_model_path}")

    # 绘制训练历史
    plot_training_history(train_losses, test_losses, train_accs, test_accs, config['save_dir'])

    # 测试最终模型
    print("\n在测试集上评估最终模型...")
    model.load_state_dict(torch.load(final_model_path))
    test_loss, test_acc, test_f1 = test(model, test_loader, criterion)
    print(f"最终模型测试准确率: {test_acc:.4f}")

    # 绘制混淆矩阵
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output, _ = model(data)
            _, predicted = torch.max(output.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)

    # 获取类别名称
    reverse_mapping = {v: f"{k * 100:.0f}%" for k, v in CARAGEENAN_MAPPING.items()}
    class_names = [reverse_mapping[i] for i in range(9)]

    # 打印混淆矩阵
    print("\n最终混淆矩阵:")
    print(cm)

    # 绘制并保存混淆矩阵
    plot_confusion_matrix(cm, classes=class_names, save_dir=config['save_dir'], normalize=False, title='混淆矩阵')
    plot_confusion_matrix(cm, classes=class_names, save_dir=config['save_dir'], normalize=True, title='混淆矩阵')

    # 保存分类报告
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    report_path = os.path.join(config['save_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"分类报告已保存到: {report_path}")

    # 保存混淆矩阵数据
    cm_path = os.path.join(config['save_dir'], 'confusion_matrix.csv')
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
    print(f"混淆矩阵数据已保存到: {cm_path}")

    print("\n训练和评估完成!")
    print("=" * 50)
    print(f"所有结果保存在: {config['save_dir']}")
    print("包括:")
    print(f"1. 模型文件 (.pth)")
    print(f"2. 训练历史图 (training_history.png)")
    print(f"3. 混淆矩阵图 (confusion_matrix.png 和 confusion_matrix_normalized.png)")
    print(f"4. 分类报告 (classification_report.txt)")
    print(f"5. 混淆矩阵数据 (confusion_matrix.csv)")