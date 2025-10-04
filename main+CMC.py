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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import joblib
import time
from tqdm import tqdm


# ==================================================
# 1. 谱数据处理与伪图生成（修改为每个样本生成伪图）
# ==================================================
def generate_pseudo_images_per_sample():
    """为每个样本生成伪生理图"""
    print("正在读取光谱数据...")
    # 读取Excel数据
    try:
        df = pd.read_excel('D:/谱/BeefClassifier/data/zq_spectra_data.xlsx')
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return {}

    # 确保目录存在
    os.makedirs('pseudo_images/zq_CMC', exist_ok=True)
    print(f"创建伪图目录: pseudo_images/zq_CMC")

    # 定义频率和特征
    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    # 存储样本ID和浓度映射
    sample_id_map = {}
    print(f"发现 {len(df)} 个样本，开始生成伪生理图...")

    # 为每个样本生成伪图
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成伪生理图"):
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
            print(f"错误：找不到列 {e}，跳过样本 {sample_id}")
            continue

        # 归一化处理
        scaler = MinMaxScaler()
        normalized_matrix = scaler.fit_transform(feature_matrix)

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
        image_path = f'pseudo_images/zq_CMC/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 存储样本映射关系
        sample_id_map[sample_id] = {
            'concentration': conc_percent,
            'image_path': image_path
        }

    print(f'成功生成 {len(sample_id_map)} 张伪生理图')
    return sample_id_map


# ==================================================
# 2. 特征提取器（修改：冻结更多层）
# ==================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet50模型
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # 冻结更多层参数（只解冻最后2层）
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False

    def forward(self, x):
        features = self.resnet(x)
        return features.squeeze()


# ==================================================
# 3. 跨模态物理一致性模块（修改：增加dropout率）
# ==================================================
class CrossModalConsistency(nn.Module):
    def __init__(self, pseudo_feat_size, rgb_feat_size, num_classes):
        """
        跨模态物理一致性模块
        :param pseudo_feat_size: 伪生理图特征维度 (2048)
        :param rgb_feat_size: RGB图像特征维度 (2048)
        :param num_classes: 类别数量
        """
        super().__init__()

        # 物理属性编码器 (从伪生理图特征预测物理属性)
        self.phys_encoder = nn.Sequential(
            nn.Linear(pseudo_feat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 预测6个物理属性
        )

        # 物理属性解码器 (从物理属性重建伪生理图特征)
        self.phys_decoder = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, pseudo_feat_size)
        )

        # 特征融合层（增加dropout率）
        self.fusion_layer = nn.Sequential(
            nn.Linear(pseudo_feat_size + rgb_feat_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.7)  # 从0.5增加到0.7
        )

        # 精炼分类器（增加dropout率）
        self.refined_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 从0.3增加到0.5
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, pseudo_feats, rgb_feats):
        """
        前向传播
        :param pseudo_feats: 伪生理图特征 [batch_size, pseudo_feat_size]
        :param rgb_feats: RGB图像特征 [batch_size, rgb_feat_size]
        :return:
            fused_features: 融合特征
            phys_properties: 预测的物理属性
            consistency_loss: 一致性损失
            class_logits: 分类logits
        """
        # 1. 编码物理属性
        phys_properties = self.phys_encoder(pseudo_feats)

        # 2. 从物理属性重建伪生理图特征
        reconstructed_pseudo = self.phys_decoder(phys_properties)

        # 3. 计算一致性损失（重建损失）
        consistency_loss = F.mse_loss(reconstructed_pseudo, pseudo_feats)

        # 4. 特征融合
        combined_feats = torch.cat((pseudo_feats, rgb_feats), dim=1)
        fused_features = self.fusion_layer(combined_feats)

        # 5. 分类预测
        class_logits = self.refined_classifier(fused_features)

        return fused_features, phys_properties, consistency_loss, class_logits


# ==================================================
# 4. 数据集类（处理双模态特征）
# ==================================================
class DualFeatureDataset(Dataset):
    def __init__(self, pseudo_features, rgb_features, labels):
        self.pseudo_features = pseudo_features
        self.rgb_features = rgb_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.pseudo_features[idx], self.rgb_features[idx]), self.labels[idx]


# ==================================================
# 5. 保存数据集函数
# ==================================================
def save_dataset(pseudo_features, rgb_features, labels, file_path):
    """保存数据集到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save({
        'pseudo_features': pseudo_features,
        'rgb_features': rgb_features,
        'labels': labels
    }, file_path)
    print(f"数据集已保存到: {file_path}")


# ==================================================
# 6. 主流程
# ==================================================
if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 步骤1: 为每个样本生成伪图
    print("\n" + "=" * 50)
    print("阶段1: 生成伪生理图")
    print("=" * 50)
    start_time = time.time()
    sample_id_map = generate_pseudo_images_per_sample()
    print(f"伪图生成完成，耗时: {time.time() - start_time:.2f}秒")

    # 步骤2: 初始化特征提取器
    print("\n" + "=" * 50)
    print("阶段2: 特征提取")
    print("=" * 50)
    start_time = time.time()
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    print(f"特征提取器初始化完成，模型结构:")
    print(feature_extractor)

    # 步骤3: 提取所有样本的特征向量
    all_pseudo_features = []  # 伪生理图特征
    all_rgb_features = []  # RGB图像特征
    all_labels = []

    # 创建样本ID到RGB图像路径的映射
    rgb_image_map = {}
    concentrations = set()

    # 扫描RGB图像目录
    image_base_dir = 'D:/谱/BeefClassifier/data/zq_beef_images_resized'
    print(f"扫描RGB图像目录: {image_base_dir}")

    if not os.path.exists(image_base_dir):
        print(f"错误: RGB图像目录不存在: {image_base_dir}")
        exit(1)

    dir_count = 0
    img_count = 0

    for conc_dir in os.listdir(image_base_dir):
        conc_path = os.path.join(image_base_dir, conc_dir)
        if os.path.isdir(conc_path):
            dir_count += 1
            try:
                concentration = int(conc_dir)  # 尝试转换为整数
                concentrations.add(concentration)
            except ValueError:
                print(f"警告: 无法将目录名 '{conc_dir}' 转换为整数浓度值，跳过")
                continue

            for img_file in os.listdir(conc_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_count += 1
                    sample_id = f"sample_{len(rgb_image_map)}"
                    rgb_image_map[sample_id] = {
                        'image_path': os.path.join(conc_path, img_file),
                        'concentration': concentration
                    }

    print(f"发现 {dir_count} 个浓度目录, {img_count} 张RGB图像")
    print(f"检测到的浓度值: {sorted(concentrations)}")

    # 创建标签编码器，将浓度值映射为0开始的连续整数
    label_encoder = LabelEncoder()
    all_concentrations = sorted(concentrations)
    label_encoder.fit(all_concentrations)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"标签编码映射: {label_mapping}")

    # 提取特征
    print("\n开始提取特征...")
    valid_samples = 0

    for sample_id, rgb_info in tqdm(rgb_image_map.items(), total=len(rgb_image_map), desc="提取特征"):
        if sample_id not in sample_id_map:
            continue

        # 提取伪图特征
        pseudo_path = sample_id_map[sample_id]['image_path']
        if not os.path.exists(pseudo_path):
            continue

        try:
            pseudo_img = Image.open(pseudo_path).convert('RGB')
            pseudo_tensor = transform(pseudo_img).unsqueeze(0).to(device)

            with torch.no_grad():
                pseudo_features = feature_extractor(pseudo_tensor).cpu().numpy()
        except Exception as e:
            continue

        # 提取RGB图像特征
        rgb_path = rgb_info['image_path']
        if not os.path.exists(rgb_path):
            continue

        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
            rgb_tensor = transform(rgb_img).unsqueeze(0).to(device)

            with torch.no_grad():
                rgb_features = feature_extractor(rgb_tensor).cpu().numpy()
        except Exception as e:
            continue

        # 分别存储两种特征
        all_pseudo_features.append(pseudo_features)
        all_rgb_features.append(rgb_features)

        # 使用标签编码器转换标签
        label = label_encoder.transform([rgb_info['concentration']])[0]
        all_labels.append(label)
        valid_samples += 1

    # 检查是否有样本
    if valid_samples == 0:
        print("错误: 没有提取到任何样本特征!")
        exit(1)

    print(f"成功提取 {valid_samples} 个样本的特征")
    print(f"特征提取完成，耗时: {time.time() - start_time:.2f}秒")

    # 转换为PyTorch张量
    pseudo_features_tensor = torch.tensor(np.array(all_pseudo_features), dtype=torch.float32)
    rgb_features_tensor = torch.tensor(np.array(all_rgb_features), dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 打印特征维度
    print(f"伪生理图特征维度: {pseudo_features_tensor.shape}")
    print(f"RGB特征维度: {rgb_features_tensor.shape}")

    # 打印标签分布
    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    print("\n标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 个样本 (浓度: {label_encoder.inverse_transform([label])[0]}%)")

    # 步骤4: 划分数据集 (70%训练, 30%验证)
    print("\n" + "=" * 50)
    print("阶段3: 数据集划分与预处理")
    print("=" * 50)

    # 获取样本索引
    indices = np.arange(len(labels_tensor))

    # 分层抽样确保各类别比例一致
    X_train, X_val, y_train, y_val = train_test_split(
        indices, labels_tensor.numpy(), test_size=0.3,
        random_state=42, stratify=labels_tensor.numpy()
    )

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

    # 创建数据集对象
    train_dataset = DualFeatureDataset(
        pseudo_features_tensor[X_train],
        rgb_features_tensor[X_train],
        labels_tensor[X_train]
    )

    val_dataset = DualFeatureDataset(
        pseudo_features_tensor[X_val],
        rgb_features_tensor[X_val],
        labels_tensor[X_val]
    )

    # 保存数据集
    save_dataset(
        pseudo_features_tensor[X_train],
        rgb_features_tensor[X_train],
        labels_tensor[X_train],
        'datasets/zq_CMC/train_dataset.pth'
    )

    save_dataset(
        pseudo_features_tensor[X_val],
        rgb_features_tensor[X_val],
        labels_tensor[X_val],
        'datasets/zq_CMC/val_dataset.pth'
    )

    # 保存标签编码器
    os.makedirs('models/zq_CMC', exist_ok=True)
    joblib.dump(label_encoder, 'models/zq_CMC/label_encoder.pkl')
    print("标签编码器已保存到 models/zq_CMC/label_encoder.pkl")

    # 保存标签映射信息
    os.makedirs('datasets/zq_CMC', exist_ok=True)
    with open('datasets/zq_CMC/label_mapping.txt', 'w') as f:
        for original, encoded in label_mapping.items():
            f.write(f"{original}% -> {encoded}\n")
    print("标签映射信息已保存到 datasets/zq_CMC/label_mapping.txt")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

    # 步骤5: 初始化跨模态一致性模型
    print("\n" + "=" * 50)
    print("阶段4: 模型训练")
    print("=" * 50)

    num_classes = len(label_encoder.classes_)
    pseudo_feat_size = pseudo_features_tensor.shape[1]
    rgb_feat_size = rgb_features_tensor.shape[1]

    print(f"伪生理图特征维度: {pseudo_feat_size}, RGB特征维度: {rgb_feat_size}, 类别数: {num_classes}")

    model = CrossModalConsistency(
        pseudo_feat_size=pseudo_feat_size,
        rgb_feat_size=rgb_feat_size,
        num_classes=num_classes
    ).to(device)

    print("模型结构:")
    print(model)

    # 损失函数和优化器
    class_criterion = nn.CrossEntropyLoss()

    # 修改1: 增大学习率 (0.01 → 0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

    # 修改2: 使用更激进的学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.95, patience=1, verbose=True  # factor从0.9→0.95, patience从2→1
    )

    # 训练模型
    best_val_acc = 0.0

    # 修改3: 增大一致性损失权重 (2.0 → 3.0)
    lambda_consistency = 3.0

    # 修改4: 减少训练轮数 (50 → 30)
    num_epochs = 50

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'consistency_loss': []
    }

    print("\n开始训练模型...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_consistency_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        for (pseudo_feats, rgb_feats), labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            pseudo_feats = pseudo_feats.to(device, non_blocking=True)
            rgb_feats = rgb_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 前向传播
            _, _, consistency_loss, class_logits = model(pseudo_feats, rgb_feats)

            # 计算分类损失
            class_loss = class_criterion(class_logits, labels)

            # 总损失 = 分类损失 + 一致性损失
            total_loss = class_loss + lambda_consistency * consistency_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            train_loss += class_loss.item()
            train_consistency_loss += consistency_loss.item()

            # 计算准确率
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 计算训练指标
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_consistency_loss = train_consistency_loss / len(train_loader)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for (pseudo_feats, rgb_feats), labels in val_loader:
                pseudo_feats = pseudo_feats.to(device, non_blocking=True)
                rgb_feats = rgb_feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # 前向传播
                _, _, _, class_logits = model(pseudo_feats, rgb_feats)

                # 计算分类损失
                loss = class_criterion(class_logits, labels)
                val_loss += loss.item()

                # 计算准确率
                _, predicted = class_logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 计算验证指标
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(val_acc)

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['consistency_loss'].append(avg_consistency_loss)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]: {epoch_time:.2f}秒")
        print(
            f"  训练损失: {avg_train_loss:.4f} | 一致性损失: {avg_consistency_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'pseudo_feat_size': pseudo_feat_size,
                'rgb_feat_size': rgb_feat_size,
                'num_classes': num_classes,
                'label_encoder': label_encoder,
                'val_acc': val_acc,
                'history': history
            }, 'models/zq_CMC/best_consistency_model.pth')
            print(f"  保存最佳模型，验证准确率: {val_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")
    print(f"最终最佳验证准确率: {best_val_acc:.2f}%")

    # 加载最佳模型
    print("\n" + "=" * 50)
    print("阶段5: 模型评估与保存")
    print("=" * 50)

    best_model = CrossModalConsistency(pseudo_feat_size, rgb_feat_size, num_classes).to(device)
    checkpoint = torch.load('models/zq_CMC/best_consistency_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()

    # 保存完整模型信息
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'pseudo_feat_size': pseudo_feat_size,
        'rgb_feat_size': rgb_feat_size,
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'val_acc': checkpoint['val_acc'],
        'history': checkpoint['history']
    }, 'models/zq_CMC/full_consistency_model.pth')
    print("完整模型已保存到 models/zq_CMC/full_consistency_model.pth")

    # 绘制训练曲线
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['consistency_loss'], label='Consistency Loss', color='red')
    plt.title('Consistency Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/zq_CMC/training_history.png')
    print("训练历史图已保存到 models/zq_CMC/training_history.png")

    # 最终验证
    final_correct = 0
    final_total = 0

    with torch.no_grad():
        for (pseudo_feats, rgb_feats), labels in val_loader:
            pseudo_feats = pseudo_feats.to(device, non_blocking=True)
            rgb_feats = rgb_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播
            _, _, _, class_logits = best_model(pseudo_feats, rgb_feats)

            # 计算准确率
            _, predicted = class_logits.max(1)
            final_total += labels.size(0)
            final_correct += predicted.eq(labels).sum().item()

    final_acc = 100. * final_correct / final_total
    print(f"最终验证准确率: {final_acc:.2f}%")

    print("\n" + "=" * 50)
    print("跨模态物理一致性牛肉质量分类系统训练完成!")
    print("=" * 50)