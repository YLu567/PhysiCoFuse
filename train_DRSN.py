import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入您的DRSN模型
from model.DRSN import DRSN_ResNet_6, BasicBlock  # 根据实际文件名修改

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 标签映射
CARAGEENAN_MAPPING = {
    0.0: 0, 0.03: 1, 0.05: 2, 0.08: 3,
    0.10: 4, 0.12: 5, 0.15: 6, 0.17: 7, 0.20: 8
}

# 频率点
FREQUENCIES = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]

# 特征列表
FEATURES = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']


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


def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    print(f"Data loaded. Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")

    if '卡拉胶含量' in df.columns:
        labels = df['卡拉胶含量'].map(CARAGEENAN_MAPPING).values
        df = df.drop(columns=['卡拉胶含量'])
    else:
        raise ValueError("'卡拉胶含量' column not found in the data")

    n_samples = len(df)
    n_freqs = len(FREQUENCIES)
    n_features = len(FEATURES)

    data = np.zeros((n_samples, n_freqs, n_features))

    for sample_idx in range(n_samples):
        for freq_idx, freq in enumerate(FREQUENCIES):
            for feat_idx, feat in enumerate(FEATURES):
                col_name = f"{feat}_{freq}"
                if col_name in df.columns:
                    data[sample_idx, freq_idx, feat_idx] = df.loc[sample_idx, col_name]
                else:
                    if feat == 'Φ':
                        col_name = f"Phi_{freq}"
                        if col_name in df.columns:
                            data[sample_idx, freq_idx, feat_idx] = df.loc[sample_idx, col_name]
                    else:
                        print(f"Warning: Column {feat}_{freq} not found. Using 0.")

    scalers = {}
    for feat_idx in range(n_features):
        feat_data = data[:, :, feat_idx].reshape(-1, 1)
        scaler = StandardScaler()
        feat_data_scaled = scaler.fit_transform(feat_data)
        data[:, :, feat_idx] = feat_data_scaled.reshape(n_samples, n_freqs)
        scalers[feat_idx] = scaler

    # 调整数据形状以适应DRSN模型
    data = data.transpose(0, 2, 1)  # 变为 (n_samples, n_features, n_freqs)

    return data, labels, scalers


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # DRSN模型期望输入形状为 (batch_size, 1, seq_length)
        batch_size, n_features, seq_length = data.shape
        data = data.view(batch_size * n_features, 1, seq_length)

        # 前向传播
        output = model(data)

        # 平均所有特征通道的输出
        output = output.view(batch_size, n_features, -1).mean(dim=1)

        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f'\nTrain set: Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1: {f1:.4f}')

    return avg_loss, acc


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 调整输入形状以适应模型
            batch_size, n_features, seq_length = data.shape
            data = data.view(batch_size * n_features, 1, seq_length)

            output = model(data)
            output = output.view(batch_size, n_features, -1).mean(dim=1)  # 平均所有特征通道的输出

            total_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_targets, all_predictions)  # 计算准确率
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f'\nTest set: Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1: {f1:.4f}')

    cm = confusion_matrix(all_targets, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    from sklearn.metrics import classification_report
    reverse_mapping = {v: f"{k * 100:.0f}%" for k, v in CARAGEENAN_MAPPING.items()}
    target_names = [reverse_mapping[i] for i in range(9)]

    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=target_names))

    return avg_loss, acc, f1  # 返回准确率


def plot_training_history(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')  # 绘制验证准确率曲线
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    # 配置参数
    config = {
        'data_path': 'D:/谱/BeefClassifier/data/zq_spectra_data.xlsx',  # 你的谱数据文件路径
        'test_size': 0.2,  # 测试集比例
        'random_seed': 42,  # 随机种子
        'batch_size': 16,  # 批大小
        'learning_rate': 0.1,  # 学习率
        'num_epochs': 50,  # 训练轮数
        'input_channels': 1,  # DRSN模型固定为1个输入通道
        'num_classes': 9,  # 类别数(9种卡拉胶含量)
        'save_dir': 'checkpoints/DRSN',  # 模型保存目录
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 加载并预处理数据
    print("Loading and preprocessing data...")
    try:
        data, labels, scalers = load_and_preprocess_data(config['data_path'])
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
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

    print(f"Data loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    # 初始化模型
    model = DRSN_ResNet_6(num_classes=config['num_classes']).to(device)

    # 打印模型结构
    print("\nModel architecture:")
    print(model)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M, Trainable: {trainable_params / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.95, patience=3, verbose=True
    )

    # 训练循环
    best_f1 = 0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []  # 用于存储训练和验证准确率

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")

        # 训练
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)

        # 验证（测试验证集）
        test_loss, test_acc, test_f1 = test(model, test_loader, criterion)  # 获取验证集准确率

        # 记录指标
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)  # 保存训练准确率
        test_accs.append(test_acc)  # 保存验证准确率

        # 更新学习率
        scheduler.step(test_f1)

        # 保存最佳模型
        if test_f1 > best_f1:
            best_f1 = test_f1
            model_path = os.path.join(config['save_dir'], f'best_drsn_epoch{epoch}_f1_{test_f1:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_f1': test_f1,
                'config': config,
            }, model_path)
            print(f"Saved best model to {model_path} with F1 {test_f1:.4f}")

    # 训练结束
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time // 60:.0f}m {training_time % 60:.0f}s")
    print(f"Best validation F1: {best_f1:.4f}")

    # 保存最终模型
    final_model_path = os.path.join(config['save_dir'], 'final_drsn_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # 绘制训练曲线（包括准确率曲线）
    plot_training_history(train_losses, test_losses, train_accs, test_accs)

    # 测试最终模型（在验证集上）
    print("\nTesting final model on validation set...")
    model.load_state_dict(torch.load(final_model_path))
    test_loss, test_acc, test_f1 = test(model, test_loader, criterion)
    print(f"Final model validation accuracy: {test_acc:.4f}")

    # 绘制混淆矩阵
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 调整输入形状以适应模型
            batch_size, n_features, seq_length = data.shape
            data = data.view(batch_size * n_features, 1, seq_length)

            output = model(data)
            output = output.view(batch_size, n_features, -1).mean(dim=1)  # 平均所有特征通道的输出

            _, predicted = torch.max(output.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_targets, all_predictions)
    # 创建反向标签映射
    reverse_mapping = {v: f"{k * 100:.0f}%" for k, v in CARAGEENAN_MAPPING.items()}
    class_names = [reverse_mapping[i] for i in range(9)]
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix')