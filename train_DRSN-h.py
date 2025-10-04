# train_drsn_carageenan.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# -------------- 设备 --------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------- 标签映射 --------------
CARAGEENAN_MAPPING = {
    0.0: 0, 0.03: 1, 0.05: 2, 0.08: 3,
    0.10: 4, 0.12: 5, 0.15: 6, 0.17: 7, 0.20: 8
}
FREQUENCIES = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
FEATURES = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

# -------------- 数据集 --------------
class CarageenanDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -------------- 数据读取 --------------
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    print(f"Data loaded. Shape={df.shape}")

    # 处理 Phi -> Φ
    for freq in FREQUENCIES:
        old = f"Phi_{freq}"
        new = f"Φ_{freq}"
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    labels = df['卡拉胶含量'].map(CARAGEENAN_MAPPING).values
    df = df.drop(columns=['卡拉胶含量'])

    n_samples, n_freqs, n_feats = len(df), len(FREQUENCIES), len(FEATURES)
    data = np.zeros((n_samples, n_freqs, n_feats))

    for i in range(n_samples):
        for j, f in enumerate(FREQUENCIES):
            for k, feat in enumerate(FEATURES):
                col = f"{feat}_{f}"
                data[i, j, k] = df.loc[i, col] if col in df.columns else 0.0

    # 标准化
    for k in range(n_feats):
        tmp = data[:, :, k].reshape(-1, 1)
        data[:, :, k] = StandardScaler().fit_transform(tmp).reshape(n_samples, n_freqs)

    # 调整维度：(batch, features, freq) -> (batch*features, 1, freq)
    data = data.transpose(0, 2, 1)  # (batch, features, freq)
    return data, labels

# -------------- 训练 / 验证 --------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, n = 0, 0
    all_t, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        b, f, s = x.shape
        x = x.view(b * f, 1, s)

        optimizer.zero_grad()
        out = model(x).view(b, f, -1).mean(dim=1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * b
        n += b
        _, pred = torch.max(out, 1)
        all_t.extend(y.cpu().numpy())
        all_p.extend(pred.cpu().numpy())

    acc = accuracy_score(all_t, all_p)
    prec = precision_score(all_t, all_p, average='macro')
    rec = recall_score(all_t, all_p, average='macro')
    f1 = f1_score(all_t, all_p, average='macro')
    return loss_sum / n, acc, prec, rec, f1, all_t, all_p

def val_epoch(model, loader, criterion):
    model.eval()
    loss_sum, n = 0, 0
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            b, f, s = x.shape
            x = x.view(b * f, 1, s)

            out = model(x).view(b, f, -1).mean(dim=1)
            loss_sum += criterion(out, y).item() * b
            n += b
            _, pred = torch.max(out, 1)
            all_t.extend(y.cpu().numpy())
            all_p.extend(pred.cpu().numpy())

    acc = accuracy_score(all_t, all_p)
    prec = precision_score(all_t, all_p, average='macro')
    rec = recall_score(all_t, all_p, average='macro')
    f1 = f1_score(all_t, all_p, average='macro')
    return loss_sum / n, acc, prec, rec, f1, all_t, all_p

# -------------- 画图 --------------
def plot_history(train_loss, test_loss, train_acc, test_acc, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(); plt.grid()

    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练历史图已保存: {save_path}")

def plot_cm(cm, classes, save_dir, normalize=False):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True,
                fmt='.2%' if normalize else 'd',
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                cbar=True,
                square=True,
                linewidths=1,
                linecolor='gray',
                annot_kws={'size': 14})
    title = '归一化混淆矩阵' if normalize else '混淆矩阵（计数）'
    plt.title(title, fontsize=18)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    suffix = '_normalized' if normalize else '_count'
    save_path = os.path.join(save_dir, f'confusion_matrix{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

# -------------- 主程序 --------------
if __name__ == '__main__':
    # ---------------- 参数 ----------------
    cfg = {
        'data_path': 'D:/谱/BeefClassifier/data/zq_spectra_data.xlsx',
        'test_ratio': 0.2,
        'seed': 42,
        'batch': 16,
        'lr': 0.1,
        'epochs': 50,
        'save_dir': 'checkpoints/DRSN'
    }
    os.makedirs(cfg['save_dir'], exist_ok=True)
    torch.manual_seed(cfg['seed'])

    # ---------------- 数据 ----------------
    data, labels = load_and_preprocess_data(cfg['data_path'])
    dataset = CarageenanDataset(data, labels)

    n_total = len(dataset)
    n_test = int(cfg['test_ratio'] * n_total)
    n_train = n_total - n_test

    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch'], shuffle=False)

    # ---------------- 模型 ----------------
    # 假设 DRSN_ResNet_6 已正确实现
    from model.DRSN import DRSN_ResNet_6
    model = DRSN_ResNet_6(num_classes=9).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=3, verbose=True)

    # ---------------- 训练 ----------------
    best_f1 = 0
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(1, cfg['epochs'] + 1):
        tr_loss, tr_acc, _, _, tr_f1, _, _ = train_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc, _, _, te_f1, te_true, te_pred = val_epoch(model, test_loader, criterion)

        train_losses.append(tr_loss); train_accs.append(tr_acc)
        test_losses.append(te_loss);  test_accs.append(te_acc)
        scheduler.step(te_f1)

        if te_f1 > best_f1:
            best_f1 = te_f1
            torch.save({'model': model.state_dict(),
                        'epoch': epoch,
                        'cfg': cfg},
                       os.path.join(cfg['save_dir'], f'best_model_f1_{te_f1:.4f}.pth'))
            print(f"↑ 保存最佳模型，F1={te_f1:.4f}")

    # ---------------- 保存 & 画图 ----------------
    torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'final_model.pth'))
    plot_history(train_losses, test_losses, train_accs, test_accs, cfg['save_dir'])

    # 混淆矩阵
    cm = confusion_matrix(te_true, te_pred)
    class_names = [f"{k*100:.0f}%" for k in sorted(CARAGEENAN_MAPPING.keys())]

    plot_cm(cm, class_names, cfg['save_dir'], normalize=False)
    plot_cm(cm, class_names, cfg['save_dir'], normalize=True)

    report = classification_report(te_true, te_pred, target_names=class_names)
    with open(os.path.join(cfg['save_dir'], 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(cfg['save_dir'], 'confusion_matrix.csv'))

    print("\n训练与评估完成！所有结果已保存在", cfg['save_dir'])