# -*- coding: utf-8 -*-
"""
PhysiCoFuse 完整代码（加速 + 半精度模型保存）
- 使用 EIS (8×6) + RGB 图像
- 可学习 PPMG (生成 4 通道伪生理属性图)
- 7 通道输入 ResNet50 (RGB + 4 伪图)
- MIE + CMPC + 分类器
- 5 折 Stratified Cross-Validation
- 数据增强 (仅训练集)
- 进度条可视化 (tqdm)
- 冻结 ResNet 浅层 (加速)
- 混合精度训练 (若 GPU 可用)
- 加速设置：较大 batch size、cudnn.benchmark、减少验证频率、prefetch
- 模型保存为半精度 (float16)，显著减小 pth 文件大小
"""

import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler  # 添加 MinMaxScaler
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 32               # 增大 batch size（显存不足可调为 24 或 16）
EPOCHS = 50
LR_MAIN = 1e-3                # 分类器 / ResNet 学习率（微调）
LR_OTHER = 2e-4               # MIE, PPMG, CMPC 学习率
WEIGHT_DECAY = 1e-4
PATIENCE = 10                  # 早停耐心
CONSISTENCY_WEIGHT = 0.01
VALIDATION_INTERVAL = 2       # 每 2 个 epoch 验证一次，减少验证开销

# 数据目录（修改为您的实际路径）
DATA_ROOT = './data'
SPECTRA_DIR = os.path.join(DATA_ROOT, 'spectra')
LABELS_DIR = os.path.join(DATA_ROOT, 'labels')
IMAGES_DIR = os.path.join(DATA_ROOT, 'images')

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 固定频率和特征名（与 Excel 列名一致）
FREQUENCIES = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
FEATURES = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关闭确定性以启用 cuDNN 自动优化（提升速度）
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed()

# ====================== 数据加载 ======================
def load_data_from_directories(spectra_dir, labels_dir, images_dir):
    xlsx_files = [f for f in os.listdir(spectra_dir) if f.lower().endswith('.xlsx')]
    if not xlsx_files:
        raise FileNotFoundError(f"在 {spectra_dir} 中未找到任何 .xlsx 文件")
    sample_ids = [os.path.splitext(f)[0] for f in xlsx_files]
    print(f"找到 {len(sample_ids)} 个样本")

    rgb_paths = []
    eis_matrices = []
    concentrations = []

    for sid in tqdm(sample_ids, desc="加载样本"):
        # 标签
        label_file = os.path.join(labels_dir, sid + '.txt')
        if not os.path.exists(label_file):
            continue
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            match = re.search(r'(\d+(\.\d+)?)%?', content)
            conc = float(match.group(1)) if match else float(content)
            concentration = int(conc)
        except:
            continue

        # EIS
        spec_file = os.path.join(spectra_dir, sid + '.xlsx')
        try:
            df = pd.read_excel(spec_file)
            mat = np.zeros((len(FREQUENCIES), len(FEATURES)), dtype=np.float32)
            for i, freq in enumerate(FREQUENCIES):
                for j, feat in enumerate(FEATURES):
                    col_name = f"{feat}_{freq}"
                    if col_name not in df.columns:
                        col_name2 = f"{feat}{freq}"
                        if col_name2 in df.columns:
                            val = df[col_name2].iloc[0]
                        else:
                            raise KeyError
                    else:
                        val = df[col_name].iloc[0]
                    mat[i, j] = float(val)
        except:
            continue

        # 图像
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(images_dir, sid + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        rgb_paths.append(img_path)
        eis_matrices.append(mat)
        concentrations.append(concentration)

    if len(rgb_paths) == 0:
        raise RuntimeError("未成功加载任何样本，请检查数据")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(concentrations)
    print(f"浓度类别: {label_encoder.classes_}")
    print(f"总样本数: {len(rgb_paths)}")
    return rgb_paths, eis_matrices, labels, label_encoder

# ====================== 数据集类 ======================
class PhysiCoFuseDataset(Dataset):
    def __init__(self, rgb_paths, eis_matrices, labels, transform=None):
        self.rgb_paths = rgb_paths
        self.eis_matrices = eis_matrices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        if self.transform:
            rgb = self.transform(rgb)
        else:
            rgb = transforms.ToTensor()(rgb)
        eis = torch.tensor(self.eis_matrices[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgb, eis, label

# ====================== 模型组件（论文架构，完全不变） ======================
class MultiFrequencyImpedanceEncoder(nn.Module):
    def __init__(self, num_freq=8, num_params=6, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(num_params, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.final_freq_dim = num_freq // 4
        self.freq_attention = nn.Sequential(
            nn.Linear(self.final_freq_dim, self.final_freq_dim),
            nn.ReLU(),
            nn.Linear(self.final_freq_dim, self.final_freq_dim),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feat_dim)
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        attn_in = x.mean(dim=1)
        attn_weights = self.freq_attention(attn_in)
        attn_weights = attn_weights.unsqueeze(1)
        x = x * attn_weights
        x = x.mean(dim=2)
        return self.fc(x), attn_weights.squeeze(1)

class PseudoBiophysicalMapGenerator(nn.Module):
    def __init__(self, imp_feat_dim=128, num_attrs=4, map_size=224):
        super().__init__()
        self.num_attrs = num_attrs
        self.map_size = map_size
        self.fc1 = nn.Linear(imp_feat_dim, 256)
        self.fc2 = nn.Linear(256, num_attrs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, imp_feat):
        h = F.relu(self.fc1(imp_feat))
        attr_scores = self.sigmoid(self.fc2(h))
        attr_maps = attr_scores.view(-1, self.num_attrs, 1, 1)
        attr_maps = attr_maps.expand(-1, -1, self.map_size, self.map_size)
        return attr_maps, attr_scores

class ResNet50_7ch(nn.Module):
    def __init__(self, freeze_layers=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(7, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        resnet.conv1 = new_conv
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 2048

        if freeze_layers:
            for i, child in enumerate(self.features.children()):
                if i < 5:   # 前5个模块 (conv1 ~ layer3)
                    for param in child.parameters():
                        param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x.squeeze(-1).squeeze(-1)

class CrossModalPhysicalConsistency(nn.Module):
    def __init__(self, img_feat_dim=2048, imp_feat_dim=128, num_attrs=4, fused_dim=512):
        super().__init__()
        self.num_attrs = num_attrs
        self.img_to_phys = nn.Sequential(
            nn.Linear(img_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_attrs)
        )
        self.imp_to_phys = nn.Sequential(
            nn.Linear(imp_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_attrs)
        )
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + imp_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU()
        )
        self.fused_dim = fused_dim

    def forward(self, img_feat, imp_feat):
        img_phys = self.img_to_phys(img_feat)
        imp_phys = self.imp_to_phys(imp_feat)
        consistency_loss = F.mse_loss(img_phys, imp_phys)
        fused = torch.cat([img_feat, imp_feat], dim=1)
        fused = self.fusion(fused)
        return fused, consistency_loss

class FeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class PhysiCoFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mie = MultiFrequencyImpedanceEncoder()
        self.ppmg = PseudoBiophysicalMapGenerator(imp_feat_dim=self.mie.feat_dim)
        self.resnet = ResNet50_7ch(freeze_layers=True)
        self.cmpc = CrossModalPhysicalConsistency(
            img_feat_dim=self.resnet.out_dim,
            imp_feat_dim=self.mie.feat_dim
        )
        self.classifier = FeatureClassifier(
            input_dim=self.cmpc.fused_dim,
            num_classes=num_classes
        )

    def forward(self, rgb, eis, return_consistency=False):
        imp_feat, _ = self.mie(eis)
        pseudo_maps, _ = self.ppmg(imp_feat)
        combined = torch.cat([rgb, pseudo_maps], dim=1)
        img_feat = self.resnet(combined)
        fused_feat, consistency_loss = self.cmpc(img_feat, imp_feat)
        logits = self.classifier(fused_feat)
        if return_consistency:
            return logits, consistency_loss
        return logits

# ====================== 训练与验证 ======================
def train_one_epoch(model, loader, optimizer, criterion, consistency_weight, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} 训练', leave=False, ncols=100)
    for rgb, eis, labels in pbar:
        rgb, eis, labels = rgb.to(device), eis.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, cons_loss = model(rgb, eis, return_consistency=True)
                class_loss = criterion(logits, labels)
                loss = class_loss + consistency_weight * cons_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, cons_loss = model(rgb, eis, return_consistency=True)
            class_loss = criterion(logits, labels)
            loss = class_loss + consistency_weight * cons_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        current_acc = 100. * correct / total
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def validate(model, loader, criterion, consistency_weight):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for rgb, eis, labels in tqdm(loader, desc='验证', leave=False, ncols=100):
            rgb, eis, labels = rgb.to(device), eis.to(device), labels.to(device)
            logits, cons_loss = model(rgb, eis, return_consistency=True)
            loss = criterion(logits, labels) + consistency_weight * cons_loss
            total_loss += loss.item()
            _, pred = logits.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

# ====================== 主程序 ======================
def main():
    print("加载数据...")
    rgb_paths, eis_matrices, labels, label_encoder = load_data_from_directories(
        SPECTRA_DIR, LABELS_DIR, IMAGES_DIR
    )
    num_classes = len(label_encoder.classes_)

    # 保存原始EIS矩阵（用于生成热图，与第二段代码一致）
    raw_eis_matrices = eis_matrices  # 原始未标准化矩阵

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_accs = []

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for fold, (train_idx, val_idx) in enumerate(skf.split(rgb_paths, labels)):
        print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")
        train_rgb = [rgb_paths[i] for i in train_idx]
        val_rgb = [rgb_paths[i] for i in val_idx]
        train_eis = [eis_matrices[i] for i in train_idx]
        val_eis = [eis_matrices[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        # 标准化 EIS
        train_eis_flat = np.array([mat.flatten() for mat in train_eis])
        val_eis_flat = np.array([mat.flatten() for mat in val_eis])
        scaler_eis = StandardScaler()
        train_eis_scaled = scaler_eis.fit_transform(train_eis_flat)
        val_eis_scaled = scaler_eis.transform(val_eis_flat)
        train_eis_scaled = train_eis_scaled.reshape(-1, 8, 6)
        val_eis_scaled = val_eis_scaled.reshape(-1, 8, 6)

        train_dataset = PhysiCoFuseDataset(train_rgb, train_eis_scaled, train_labels, transform=train_transform)
        val_dataset = PhysiCoFuseDataset(val_rgb, val_eis_scaled, val_labels, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True, prefetch_factor=2)

        model = PhysiCoFuse(num_classes).to(device)

        optimizer = optim.Adam([
            {'params': model.mie.parameters(), 'lr': LR_OTHER},
            {'params': model.ppmg.parameters(), 'lr': LR_OTHER},
            {'params': model.resnet.parameters(), 'lr': LR_MAIN},
            {'params': model.cmpc.parameters(), 'lr': LR_OTHER},
            {'params': model.classifier.parameters(), 'lr': LR_MAIN}
        ], weight_decay=WEIGHT_DECAY)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                                     CONSISTENCY_WEIGHT, epoch, scaler)

            # 每隔 VALIDATION_INTERVAL 个 epoch 验证一次，且最后 epoch 必须验证
            if (epoch + 1) % VALIDATION_INTERVAL == 0 or (epoch + 1) == EPOCHS:
                val_loss, val_acc = validate(model, val_loader, criterion, CONSISTENCY_WEIGHT)
                scheduler.step(val_loss)
                print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}% (Loss {val_loss:.4f})")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # ========== 保存半精度模型 ==========
                    half_state_dict = {k: v.half() for k, v in model.state_dict().items()}
                    torch.save(half_state_dict, os.path.join(OUTPUT_DIR, f'best_model_fold{fold+1}.pth'))
                    print(f"  ✅ 保存最佳模型 (半精度, Val Acc {val_acc:.2f}%)")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"⏹️ 早停于 epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% (skip validation)")

        fold_accs.append(best_val_acc)
        print(f"Fold {fold+1} 最佳验证准确率: {best_val_acc:.2f}%")

        # ===== 保存原始EIS热图（与第二段代码一致） =====
        print(f"  保存原始EIS热图...")
        num_save = min(4, len(val_idx))
        for i in range(num_save):
            idx = val_idx[i]
            raw_eis = raw_eis_matrices[idx]  # shape (8,6)
            conc = label_encoder.inverse_transform([labels[idx]])[0]  # 原始浓度（int）
            # 归一化（MinMax）
            scaler_vis = MinMaxScaler()
            normalized = scaler_vis.fit_transform(raw_eis)
            plt.figure(figsize=(8, 6))
            plt.imshow(normalized, cmap='viridis', aspect='auto')
            plt.colorbar(label='Normalized Value')
            plt.title(f'Sample {idx}, Carrageenan: {conc}%')
            plt.xlabel('Features')
            plt.ylabel('Frequency (Hz)')
            plt.xticks(range(len(FEATURES)), FEATURES)
            plt.yticks(range(len(FREQUENCIES)), [str(f) for f in FREQUENCIES])
            save_path = os.path.join(OUTPUT_DIR, f'eis_heatmap_fold{fold+1}_sample{i}_conc{conc}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  保存热图: {save_path}")

    avg_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print("\n========== 最终结果 ==========")
    print(f"各折准确率: {[f'{a:.2f}%' for a in fold_accs]}")
    print(f"平均准确率: {avg_acc:.2f}% ± {std_acc:.2f}%")

    with open(os.path.join(OUTPUT_DIR, 'cv_results.txt'), 'w') as f:
        f.write(f"5-Fold CV Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Per fold: {fold_accs}\n")

if __name__ == '__main__':
    main()
