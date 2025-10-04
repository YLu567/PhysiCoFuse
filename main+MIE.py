import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import joblib


# ==================================================
# 1. 谱数据处理与伪图生成（修改为保存原始谱数据）
# ==================================================
def generate_pseudo_images_per_sample():
    # 读取Excel数据
    df = pd.read_excel('D:/谱/BeefClassifier/data/zq_spectra_data.xlsx')

    # 确保目录存在
    os.makedirs('pseudo_images/zq_MIE', exist_ok=True)

    # 定义频率和特征
    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    # 存储样本ID和浓度映射
    sample_id_map = {}

    # 为每个样本生成伪图
    for idx, row in df.iterrows():
        sample_id = f"sample_{idx}"
        concentration = row['卡拉胶含量']
        conc_percent = int(concentration * 100)  # 转换为整数百分比

        # 创建8x6的特征矩阵 (8个频率 x 6个特征)
        feature_matrix = np.zeros((len(frequencies), len(features)))

        # 填充矩阵
        try:
            for i, freq in enumerate(frequencies):
                for j, feat in enumerate(features):
                    col_name = f"{feat}_{freq}"
                    feature_matrix[i, j] = row[col_name]
        except KeyError as e:
            print(f"错误：找不到列 {e}，检查列名是否匹配")
            continue

        # 归一化处理
        scaler = MinMaxScaler()
        normalized_matrix = scaler.fit_transform(feature_matrix)

        # 保存原始谱数据（归一化后的）
        sample_id_map[sample_id] = {
            'concentration': conc_percent,
            'raw_matrix': normalized_matrix.copy()  # 保存归一化后的原始矩阵
        }

        # 创建伪生理图
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Normalized Value')
        plt.title(f'Sample ID: {sample_id}, Carrageenan: {conc_percent}%')
        plt.xlabel('Features')
        plt.ylabel('Frequency (Hz)')
        plt.xticks(range(len(features)), features)
        plt.yticks(range(len(frequencies)), [str(f) for f in frequencies])

        # 保存伪图（使用样本ID作为文件名）
        image_path = f'pseudo_images/zq_MIE/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 更新映射关系
        sample_id_map[sample_id]['image_path'] = image_path

        print(f'Generated pseudo image: {image_path}')

    return sample_id_map


# ==================================================
# 2. 特征提取器
# ==================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        return self.resnet(x).squeeze()


# ==================================================
# 3. 多频率阻抗编码器 (MIE)
# ==================================================
class MultiFrequencyImpedanceEncoder(nn.Module):
    def __init__(self, num_freq=8, num_params=6, feature_dim=128):
        super().__init__()
        self.num_freq = num_freq
        self.num_params = num_params
        self.feature_dim = feature_dim

        # 1D卷积层提取频率特征
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_params, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        # 频率注意力机制
        self.freq_attention = nn.Sequential(
            nn.Linear(num_freq, 32),
            nn.ReLU(),
            nn.Linear(32, num_freq),
            nn.Softmax(dim=1)
        )

        # 特征融合层 - 修复输入维度
        self.fusion = nn.Sequential(
            nn.Linear(128 + num_params, 256),  # 输入维度应为128+6=134
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        # x形状: [batch, num_freq, num_params]

        # 转换为卷积需要的形状: [batch, num_params, num_freq]
        x_conv = x.permute(0, 2, 1)

        # 1D卷积处理
        conv_features = self.conv_layers(x_conv).squeeze(-1)  # [batch, 128]

        # 频率注意力
        # 在参数维度上取平均: [batch, num_freq]
        mean_features = x.mean(dim=2)
        attn_weights = self.freq_attention(mean_features)  # [batch, num_freq]
        # 使用注意力权重对原始x加权
        attn_features = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, num_params]

        # 特征融合
        fused_features = torch.cat([conv_features, attn_features], dim=1)
        output = self.fusion(fused_features)

        # 返回编码特征和频率权重
        return output, attn_weights

# ==================================================
# 4. 数据集类（更新为包含原始谱数据）
# ==================================================
class FeatureDataset(Dataset):
    def __init__(self, pseudo_features, rgb_features, raw_matrices, labels):
        self.pseudo_features = pseudo_features
        self.rgb_features = rgb_features
        self.raw_matrices = raw_matrices
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.pseudo_features[idx],
            self.rgb_features[idx],
            self.raw_matrices[idx],
            self.labels[idx]
        )


# ==================================================
# 5. 分类器模型（更新输入尺寸）
# ==================================================
class FeatureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ==================================================
# 6. 保存数据集函数
# ==================================================
def save_dataset(dataset, file_path):
    """保存数据集到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save({
        'pseudo_features': dataset.pseudo_features,
        'rgb_features': dataset.rgb_features,
        'raw_matrices': dataset.raw_matrices,
        'labels': dataset.labels
    }, file_path)
    print(f"数据集已保存到: {file_path}")


# ==================================================
# 7. 主流程
# ==================================================
if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 步骤1: 为每个样本生成伪图
    print("正在为每个样本生成伪图...")
    sample_id_map = generate_pseudo_images_per_sample()

    # 步骤2: 初始化特征提取器
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # 步骤3: 提取所有样本的特征向量
    print("正在提取特征向量...")
    all_pseudo_features = []
    all_rgb_features = []
    all_raw_matrices = []  # 存储原始谱数据
    all_labels = []

    # 创建样本ID到RGB图像路径的映射
    rgb_image_map = {}
    concentrations = set()

    # 扫描RGB图像目录
    image_base_dir = 'D:/谱/BeefClassifier/data/zq_beef_images_resized'
    for conc_dir in os.listdir(image_base_dir):
        conc_path = os.path.join(image_base_dir, conc_dir)
        if os.path.isdir(conc_path):
            try:
                concentration = int(conc_dir)  # 尝试转换为整数
                concentrations.add(concentration)
            except ValueError:
                print(f"警告: 无法将目录名 '{conc_dir}' 转换为整数浓度值，跳过")
                continue

            for img_file in os.listdir(conc_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_id = f"sample_{len(rgb_image_map)}"
                    rgb_image_map[sample_id] = {
                        'image_path': os.path.join(conc_path, img_file),
                        'concentration': concentration
                    }

    # 打印所有检测到的浓度值
    print(f"检测到的浓度值: {sorted(concentrations)}")

    # 创建标签编码器，将浓度值映射为0开始的连续整数
    label_encoder = LabelEncoder()
    all_concentrations = sorted(concentrations)
    label_encoder.fit(all_concentrations)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"标签编码映射: {label_mapping}")

    # 提取特征
    for sample_id, rgb_info in rgb_image_map.items():
        if sample_id not in sample_id_map:
            continue

        # 提取伪图特征
        pseudo_path = sample_id_map[sample_id]['image_path']
        if not os.path.exists(pseudo_path):
            print(f"警告: 伪图文件不存在: {pseudo_path}")
            continue

        try:
            pseudo_img = Image.open(pseudo_path).convert('RGB')
            pseudo_tensor = transform(pseudo_img).unsqueeze(0).to(device)

            with torch.no_grad():
                pseudo_features = feature_extractor(pseudo_tensor).cpu().numpy()
        except Exception as e:
            print(f"处理伪图 {pseudo_path} 时出错: {e}")
            continue

        # 提取RGB图像特征
        rgb_path = rgb_info['image_path']
        if not os.path.exists(rgb_path):
            print(f"警告: RGB图像文件不存在: {rgb_path}")
            continue

        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
            rgb_tensor = transform(rgb_img).unsqueeze(0).to(device)

            with torch.no_grad():
                rgb_features = feature_extractor(rgb_tensor).cpu().numpy()
        except Exception as e:
            print(f"处理RGB图像 {rgb_path} 时出错: {e}")
            continue

        # 保存原始谱数据
        raw_matrix = sample_id_map[sample_id]['raw_matrix']

        # 使用标签编码器转换标签
        label = label_encoder.transform([rgb_info['concentration']])[0]

        all_pseudo_features.append(pseudo_features)
        all_rgb_features.append(rgb_features)
        all_raw_matrices.append(raw_matrix)
        all_labels.append(label)

    # 检查是否有样本
    if len(all_pseudo_features) == 0:
        print("错误: 没有提取到任何样本特征!")
        exit(1)

    # 转换为PyTorch张量
    pseudo_features_tensor = torch.tensor(np.array(all_pseudo_features), dtype=torch.float32)
    rgb_features_tensor = torch.tensor(np.array(all_rgb_features), dtype=torch.float32)
    raw_matrices_tensor = torch.tensor(np.array(all_raw_matrices), dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 打印标签分布
    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 个样本 (浓度: {label_encoder.inverse_transform([label])[0]}%)")

    # 步骤4: 划分数据集 (70%训练, 30%验证)
    print("划分数据集...")
    indices = np.arange(len(labels_tensor))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels_tensor.numpy()
    )

    # 创建数据集对象
    train_dataset = FeatureDataset(
        pseudo_features_tensor[train_idx],
        rgb_features_tensor[train_idx],
        raw_matrices_tensor[train_idx],
        labels_tensor[train_idx]
    )

    val_dataset = FeatureDataset(
        pseudo_features_tensor[val_idx],
        rgb_features_tensor[val_idx],
        raw_matrices_tensor[val_idx],
        labels_tensor[val_idx]
    )

    # 保存数据集
    save_dataset(train_dataset, 'datasets/zq_MIE/train_dataset.pth')
    save_dataset(val_dataset, 'datasets/zq_MIE/val_dataset.pth')

    # 保存标签编码器
    os.makedirs('models/zq_MIE', exist_ok=True)
    joblib.dump(label_encoder, 'models/zq_MIE/label_encoder.pkl')
    print("标签编码器已保存到 models/zq_MIE/label_encoder.pkl")

    # 保存标签映射信息
    os.makedirs('datasets/zq_MIE', exist_ok=True)
    with open('datasets/zq_MIE/label_mapping.txt', 'w') as f:
        for original, encoded in label_mapping.items():
            f.write(f"{original}% -> {encoded}\n")
    print("标签映射信息已保存到 datasets/zq_MIE/label_mapping.txt")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 步骤5: 初始化MIE和分类器
    num_classes = len(label_encoder.classes_)
    mie = MultiFrequencyImpedanceEncoder().to(device)

    # 输入尺寸 = 伪图特征(2048) + RGB特征(2048) + MIE特征(128)
    input_size = 2048 + 2048 + 128
    classifier = FeatureClassifier(input_size, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(mie.parameters()) + list(classifier.parameters()),
        lr=0.001,
        weight_decay=1e-5
    )

    # 训练模型
    print("开始训练模型...")
    best_val_acc = 0.0

    for epoch in range(50):
        mie.train()
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for pseudo_feats, rgb_feats, raw_mats, labels in train_loader:
            pseudo_feats = pseudo_feats.to(device)
            rgb_feats = rgb_feats.to(device)
            raw_mats = raw_mats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 通过MIE处理原始谱数据
            mie_feats, _ = mie(raw_mats)

            # 拼接所有特征
            combined_feats = torch.cat([pseudo_feats, rgb_feats, mie_feats], dim=1)

            # 分类
            outputs = classifier(combined_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # 验证
        mie.eval()
        classifier.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for pseudo_feats, rgb_feats, raw_mats, labels in val_loader:
                pseudo_feats = pseudo_feats.to(device)
                rgb_feats = rgb_feats.to(device)
                raw_mats = raw_mats.to(device)
                labels = labels.to(device)

                mie_feats, _ = mie(raw_mats)
                combined_feats = torch.cat([pseudo_feats, rgb_feats, mie_feats], dim=1)
                outputs = classifier(combined_feats)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(
            f'Epoch [{epoch + 1}/50], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models/zq_MIE', exist_ok=True)
            torch.save({
                'mie_state_dict': mie.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'models/zq_MIE/best_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    # 加载最佳模型并做最终评估
    checkpoint = torch.load('models/zq_MIE/best_model.pth')
    mie.load_state_dict(checkpoint['mie_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    mie.eval()
    classifier.eval()

    final_correct = 0
    final_total = 0

    with torch.no_grad():
        for pseudo_feats, rgb_feats, raw_mats, labels in val_loader:
            pseudo_feats = pseudo_feats.to(device)
            rgb_feats = rgb_feats.to(device)
            raw_mats = raw_mats.to(device)
            labels = labels.to(device)

            mie_feats, _ = mie(raw_mats)
            combined_feats = torch.cat([pseudo_feats, rgb_feats, mie_feats], dim=1)
            outputs = classifier(combined_feats)
            _, predicted = outputs.max(1)
            final_total += labels.size(0)
            final_correct += predicted.eq(labels).sum().item()

    final_acc = 100. * final_correct / final_total
    print(f"最终验证准确率: {final_acc:.2f}%")

    # 保存完整模型（包含架构）
    torch.save({
        'mie_state_dict': mie.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'label_encoder': label_encoder
    }, 'models/zq_MIE/zq_MIE_model.pth')
    print("完整模型已保存到 models/zq_MIE/zq_MIE_model.pth")