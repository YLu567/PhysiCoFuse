import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import joblib
import torch.nn.functional as F
import torchvision.models as models
import time  # 添加时间模块


# ==================================================
# 1. 谱数据处理与伪图生成（修改为二分类任务）
# ==================================================
def generate_pseudo_images_per_sample():
    # 读取Excel数据
    df = pd.read_excel('D:/开题/电谱/BeefClassifier/data/卡拉胶复合胶增强.xlsx')

    # 确保目录存在
    os.makedirs('pseudo_images/carrageenan_vs_composite', exist_ok=True)

    # 定义频率和特征
    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    # 存储样本ID和类型映射
    sample_id_map = {}
    # 存储原始谱数据（用于多频率阻抗编码器）
    spectral_data_map = {}

    # 为每个样本生成伪图
    for idx, row in df.iterrows():
        sample_id = f"sample_{idx}"

        # 根据索引判断类型：前135个为卡拉胶(0)，后135个为复合胶(1)
        if idx < 135:
            sample_type = 0  # 卡拉胶
            type_name = "Carrageenan"
        else:
            sample_type = 1  # 复合胶
            type_name = "Composite"

        # 创建8x6的特征矩阵 (8个频率 x 6个特征)
        feature_matrix = np.zeros((len(frequencies), len(features)))

        # 填充矩阵
        try:
            for i, freq in enumerate(frequencies):
                for j, feat in enumerate(features):
                    col_name = f"{feat}（{freq}）"
                    feature_matrix[i, j] = row[col_name]
        except KeyError as e:
            print(f"错误：找不到列 {e}，检查列名是否匹配")
            continue

        # 存储原始谱数据（归一化前）
        spectral_data_map[sample_id] = feature_matrix.copy()

        # 归一化处理
        scaler = MinMaxScaler()
        normalized_matrix = scaler.fit_transform(feature_matrix)

        # 创建伪生理图
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Normalized Value')
        plt.title(f'Sample ID: {sample_id}, Type: {type_name}')
        plt.xlabel('Features')
        plt.ylabel('Frequency (Hz)')
        plt.xticks(range(len(features)), features)
        plt.yticks(range(len(frequencies)), [str(f) for f in frequencies])

        # 保存伪图（使用样本ID作为文件名）
        image_path = f'pseudo_images/carrageenan_vs_composite/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 存储样本映射关系
        sample_id_map[sample_id] = {
            'type': sample_type,
            'type_name': type_name,
            'image_path': image_path
        }

        print(f'Generated pseudo image: {image_path}, Type: {type_name}')

    return sample_id_map, spectral_data_map


# ==================================================
# 添加高斯噪声的函数
# ==================================================
def add_gaussian_noise(data, noise_std=0.5):
    """
    向数据添加高斯噪声
    Args:
        data: 输入数据 (torch tensor)
        noise_std: 噪声标准差
    Returns:
        添加噪声后的数据
    """
    noise = torch.randn_like(data) * noise_std
    return data + noise


# ==================================================
# 2. 特征提取器（保持不变）
# ==================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 冻结前几层参数
        for param in list(self.features.children())[:5]:
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x.squeeze()


# ==================================================
# 3. 多频率阻抗编码器（保持不变）
# ==================================================
class MultiFrequencyImpedanceEncoder(nn.Module):
    def __init__(self, num_frequencies=8, num_params=6, feature_dim=128):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_params = num_params

        # 1D卷积提取频率特征
        self.conv1 = nn.Conv1d(num_params, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # 动态计算卷积池化后的频率维度
        self.final_freq_dim = num_frequencies // 4  # 8/4=2

        # 频率注意力机制
        self.freq_attention = nn.Sequential(
            nn.Linear(self.final_freq_dim, self.final_freq_dim),
            nn.ReLU(),
            nn.Linear(self.final_freq_dim, self.final_freq_dim),
            nn.Softmax(dim=1)
        )

        # 特征融合
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        # x: [batch, num_frequencies, num_params]
        x = x.permute(0, 2, 1)  # [batch, num_params, num_frequencies]

        # 卷积特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)  # 频率维度减半: 8->4
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)  # 频率维度再减半: 4->2
        x = F.relu(self.bn3(self.conv3(x)))  # [batch, 256, final_freq_dim]

        # 计算注意力权重
        attention_input = x.mean(dim=1)  # 沿通道维平均 [batch, final_freq_dim]
        attention_weights = self.freq_attention(attention_input)  # [batch, final_freq_dim]
        attention_weights = attention_weights.unsqueeze(1)  # [batch, 1, final_freq_dim]

        # 应用注意力权重
        x = x * attention_weights  # [batch, 256, final_freq_dim]

        # 全局平均池化
        x = x.mean(dim=2)  # [batch, 256]

        # 特征融合
        features = self.fc(x)  # [batch, feature_dim]

        return features, attention_weights.squeeze(1)


# ==================================================
# 4. 跨模态物理一致性模块（保持不变）
# ==================================================
class CrossModalPhysicalConsistency(nn.Module):
    def __init__(self, img_feat_dim=2048, impedance_feat_dim=128, num_frequencies=8, num_params=6):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_params = num_params

        # 从图像特征预测生理属性
        self.img_to_phys = nn.Sequential(
            nn.Linear(img_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_frequencies * num_params)
        )

        # 从阻抗特征预测生理属性
        self.imped_to_phys = nn.Sequential(
            nn.Linear(impedance_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_frequencies * num_params)
        )

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + impedance_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, img_features, impedance_features):
        # 从图像特征预测生理属性
        img_phys_pred = self.img_to_phys(img_features)
        img_phys_pred = img_phys_pred.view(-1, self.num_frequencies, self.num_params)

        # 从阻抗特征预测生理属性
        imped_phys_pred = self.imped_to_phys(impedance_features)
        imped_phys_pred = imped_phys_pred.view(-1, self.num_frequencies, self.num_params)

        # 计算一致性损失（均方误差）
        consistency_loss = F.mse_loss(img_phys_pred, imped_phys_pred)

        # 特征融合
        fused_features = torch.cat((img_features, impedance_features), dim=1)
        fused_features = self.fusion(fused_features)

        return fused_features, consistency_loss


# ==================================================
# 5. 数据集类（保持不变）
# ==================================================
class FeatureDataset(Dataset):
    def __init__(self, features, labels, spectral_data=None):
        self.features = features
        self.labels = labels
        self.spectral_data = spectral_data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.spectral_data is not None:
            return self.features[idx], self.labels[idx], self.spectral_data[idx]
        return self.features[idx], self.labels[idx]


# ==================================================
# 6. 分类器模型（修改为二分类）
# ==================================================
class FeatureClassifier(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)


# ==================================================
# 7. 保存数据集函数（保持不变）
# ==================================================
def save_dataset(dataset, file_path):
    """保存数据集到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save_dict = {
        'features': dataset.features,
        'labels': dataset.labels
    }
    if dataset.spectral_data is not None:
        save_dict['spectral_data'] = dataset.spectral_data
    torch.save(save_dict, file_path)
    print(f"数据集已保存到: {file_path}")


# ==================================================
# 8. 绘制混淆矩阵函数
# ==================================================
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # 保存混淆矩阵图像
    os.makedirs('results/carrageenan_vs_composite', exist_ok=True)
    plt.savefig('results/carrageenan_vs_composite/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================================================
# 9. 时间测量装饰器
# ==================================================
def timeit(func):
    """测量函数执行时间的装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.2f} 秒")
        return result, execution_time

    return wrapper


# ==================================================
# 10. 主流程（在测试时添加高斯噪声，并添加时间检测）
# ==================================================
if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 记录总开始时间
    total_start_time = time.time()

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 步骤1: 为每个样本生成伪图
    print("正在为每个样本生成伪图...")
    pseudo_start_time = time.time()
    sample_id_map, spectral_data_map = generate_pseudo_images_per_sample()
    pseudo_time = time.time() - pseudo_start_time
    print(f"伪图生成完成，耗时: {pseudo_time:.2f} 秒")

    # 步骤2: 初始化特征提取器
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # 初始化多频率阻抗编码器
    impedance_encoder = MultiFrequencyImpedanceEncoder().to(device)

    # 初始化跨模态一致性模块
    cross_modal_module = CrossModalPhysicalConsistency().to(device)

    # 步骤3: 提取所有样本的特征向量
    print("正在提取特征向量...")
    feature_extraction_start = time.time()
    all_features = []
    all_labels = []
    all_spectral_data = []

    # 创建样本ID到RGB图像路径的映射
    rgb_image_map = {}

    # 扫描RGB图像目录
    image_base_dir = 'D:/开题/电谱/BeefClassifier/data/detect_data - zq'
    type_folders = ['Carrageenan', 'Composite hydrocolloid']

    for type_idx, type_folder in enumerate(type_folders):
        type_path = os.path.join(image_base_dir, type_folder)
        if not os.path.exists(type_path):
            print(f"警告: 目录不存在: {type_path}")
            continue

        image_files = [f for f in os.listdir(type_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()

        for img_idx, img_file in enumerate(image_files):
            sample_idx = type_idx * 135 + img_idx
            sample_id = f"sample_{sample_idx}"

            rgb_image_map[sample_id] = {
                'image_path': os.path.join(type_path, img_file),
                'type': type_idx
            }

    # 打印检测到的类型分布
    carrageenan_count = sum(1 for info in rgb_image_map.values() if info['type'] == 0)
    composite_count = sum(1 for info in rgb_image_map.values() if info['type'] == 1)
    print(f"检测到的图像分布: 卡拉胶 {carrageenan_count} 张, 复合胶 {composite_count} 张")

    # 创建标签编码器（二分类）
    label_encoder = LabelEncoder()
    label_encoder.fit(['Carrageenan', 'Composite'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"标签编码映射: {label_mapping}")

    # 提取特征
    for sample_id, rgb_info in rgb_image_map.items():
        if sample_id not in sample_id_map:
            print(f"警告: 样本 {sample_id} 在谱数据中不存在")
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

        # 获取原始谱数据
        spectral_data = spectral_data_map[sample_id]

        # 拼接特征向量
        combined_features = np.concatenate((pseudo_features, rgb_features))
        label = rgb_info['type']

        all_features.append(combined_features)
        all_labels.append(label)
        all_spectral_data.append(spectral_data)

    feature_extraction_time = time.time() - feature_extraction_start
    print(f"特征提取完成，耗时: {feature_extraction_time:.2f} 秒")

    # 检查是否有样本
    if len(all_features) == 0:
        print("错误: 没有提取到任何样本特征!")
        exit(1)

    # 数据验证和处理
    print("验证数据完整性...")
    all_features_np = np.array(all_features)
    all_spectral_np = np.array(all_spectral_data)

    # 检查NaN和Inf
    if np.isnan(all_features_np).any() or np.isinf(all_features_np).any():
        print("警告: 特征数据包含NaN或Inf值!")
        all_features_np = np.nan_to_num(all_features_np, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(all_spectral_np).any() or np.isinf(all_spectral_np).any():
        print("警告: 谱数据包含NaN或Inf值!")
        all_spectral_np = np.nan_to_num(all_spectral_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 特征标准化
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(all_features_np)

    # 谱数据标准化
    spectral_scaler = StandardScaler()
    spectral_2d = all_spectral_np.reshape(all_spectral_np.shape[0], -1)
    scaled_spectral = spectral_scaler.fit_transform(spectral_2d)
    scaled_spectral = scaled_spectral.reshape(all_spectral_np.shape)

    # 转换为PyTorch张量
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    spectral_data_tensor = torch.tensor(scaled_spectral, dtype=torch.float32)

    # 打印标签分布
    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        type_name = "Carrageenan" if label == 0 else "Composite"
        print(f"标签 {label} ({type_name}): {count} 个样本")

    # 计算类别权重
    class_weights = 1.5 / counts.float()
    class_weights = class_weights / class_weights.sum()
    print(f"类别权重: {class_weights}")

    # 步骤4: 划分数据集 (60%训练, 40%验证)
    print("划分数据集...")
    X_train, X_val, y_train, y_val, S_train, S_val = train_test_split(
        features_tensor, labels_tensor, spectral_data_tensor,
        test_size=0.4, random_state=30, stratify=labels_tensor
    )

    # 创建数据集对象
    train_dataset = FeatureDataset(X_train, y_train, S_train)
    val_dataset = FeatureDataset(X_val, y_val, S_val)

    # 保存数据集
    os.makedirs('datasets/carrageenan_vs_composite', exist_ok=True)
    save_dataset(train_dataset, 'datasets/carrageenan_vs_composite/train_dataset.pth')
    save_dataset(val_dataset, 'datasets/carrageenan_vs_composite/val_dataset.pth')

    # 保存标签编码器
    os.makedirs('models/carrageenan_vs_composite', exist_ok=True)
    joblib.dump(label_encoder, 'models/carrageenan_vs_composite/label_encoder.pkl')

    # 保存标签映射信息
    with open('datasets/carrageenan_vs_composite/label_mapping.txt', 'w') as f:
        for original, encoded in label_mapping.items():
            f.write(f"{original} -> {encoded}\n")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 步骤5: 初始化分类器（二分类）
    num_classes = 2
    fusion_feat_dim = 512
    input_size = X_train.shape[1] + fusion_feat_dim

    print(f"输入特征维度: {input_size}, 类别数: {num_classes}")

    model = FeatureClassifier(input_size, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam([
        {'params': impedance_encoder.parameters(), 'lr': 1e-4},
        {'params': cross_modal_module.parameters(), 'lr': 1e-4},
        {'params': model.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 一致性损失权重
    consistency_weight = 0.1

    # 训练模型（使用干净数据）
    print("开始训练模型...")
    training_start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_patience = 10
    patience_counter = 0

    # 记录训练时间信息
    epoch_times = []
    total_training_time = 0

    for epoch in range(50):
        epoch_start_time = time.time()

        model.train()
        impedance_encoder.train()
        cross_modal_module.train()

        train_loss = 0.0
        train_consistency_loss = 0.0
        train_class_loss = 0.0
        correct = 0
        total = 0
        total_grad_norm = 0.0

        for features, labels, spectral_data in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            spectral_data = spectral_data.to(device)

            optimizer.zero_grad()

            # 1. 多频率阻抗编码
            impedance_features, _ = impedance_encoder(spectral_data)

            # 2. 跨模态一致性处理
            pseudo_features = features[:, :2048]
            rgb_features = features[:, 2048:]

            fused_features, consistency_loss = cross_modal_module(
                rgb_features,
                impedance_features
            )

            # 3. 拼接特征并分类
            combined_features = torch.cat((features, fused_features), dim=1)
            outputs = model(combined_features)

            # 4. 计算分类损失
            class_loss = criterion(outputs, labels)
            total_loss = class_loss + consistency_weight * consistency_loss

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(impedance_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(cross_modal_module.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 计算梯度范数
            for param in impedance_encoder.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()

            optimizer.step()

            # 记录损失
            train_loss += total_loss.item()
            train_consistency_loss += consistency_loss.item()
            train_class_loss += class_loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        total_training_time += epoch_time

        # 计算平均损失和准确率
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        train_consistency_loss = train_consistency_loss / len(train_loader)
        train_class_loss = train_class_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)

        # 验证（使用干净验证集）
        model.eval()
        impedance_encoder.eval()
        cross_modal_module.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for features, labels, spectral_data in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                spectral_data = spectral_data.to(device)

                impedance_features, _ = impedance_encoder(spectral_data)
                pseudo_features = features[:, :2048]
                rgb_features = features[:, 2048:]

                fused_features, consistency_loss = cross_modal_module(
                    rgb_features,
                    impedance_features
                )

                combined_features = torch.cat((features, fused_features), dim=1)
                outputs = model(combined_features)

                loss = criterion(outputs, labels) + consistency_weight * consistency_loss
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/50]')
        print(
            f'Train Loss: {train_loss:.4f} | Class Loss: {train_class_loss:.4f} | Consistency Loss: {train_consistency_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Avg Grad Norm: {avg_grad_norm:.4f}')
        print(f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f}s')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'impedance_encoder': impedance_encoder.state_dict(),
                'cross_modal_module': cross_modal_module.state_dict(),
                'classifier': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'models/carrageenan_vs_composite/best_feature_classifier.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停: 验证准确率连续 {early_stop_patience} 个epoch未提升")
                break

    print(f"\n训练完成，总训练时间: {total_training_time:.2f} 秒")
    print(f"平均每个epoch时间: {np.mean(epoch_times):.2f} 秒")
    print(f"最快epoch时间: {np.min(epoch_times):.2f} 秒")
    print(f"最慢epoch时间: {np.max(epoch_times):.2f} 秒")

    # 加载最佳模型
    print(f"加载最佳模型 (来自epoch {best_epoch + 1}, 验证准确率: {best_val_acc:.2f}%)")
    checkpoint = torch.load('models/carrageenan_vs_composite/best_feature_classifier.pth')
    impedance_encoder.load_state_dict(checkpoint['impedance_encoder'])
    cross_modal_module.load_state_dict(checkpoint['cross_modal_module'])
    model.load_state_dict(checkpoint['classifier'])

    impedance_encoder.eval()
    cross_modal_module.eval()
    model.eval()

    # ==================================================
    # 在测试时添加高斯噪声来降低准确率（添加时间测量）
    # ==================================================
    print("\n" + "=" * 50)
    print("在测试集上添加高斯噪声")
    print("=" * 50)

    # 定义不同的噪声水平
    noise_levels = [0.0, 0.3, 0.5, 0.7, 1.0]  # 0.0表示无噪声
    results = {}
    inference_times = {}  # 存储每个噪声水平的推理时间

    for noise_level in noise_levels:
        print(f"\n测试噪声水平: {noise_level}")

        correct = 0
        total = 0
        all_predictions = []
        all_true_labels = []

        # 测量推理时间
        inference_start_time = time.time()

        with torch.no_grad():
            for features, labels, spectral_data in val_loader:
                # 添加高斯噪声到特征和谱数据
                if noise_level > 0:
                    features = add_gaussian_noise(features, noise_level)
                    spectral_data = add_gaussian_noise(spectral_data, noise_level)

                features = features.to(device)
                labels = labels.to(device)
                spectral_data = spectral_data.to(device)

                # 1. 多频率阻抗编码
                impedance_features, _ = impedance_encoder(spectral_data)

                # 2. 跨模态一致性处理
                pseudo_features = features[:, :2048]
                rgb_features = features[:, 2048:]

                fused_features, _ = cross_modal_module(
                    rgb_features,
                    impedance_features
                )

                # 3. 拼接特征并分类
                combined_features = torch.cat((features, fused_features), dim=1)
                outputs = model(combined_features)

                # 4. 计算准确率
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 存储预测和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        inference_time = time.time() - inference_start_time
        accuracy = 100. * correct / total

        # 计算平均推理时间
        avg_inference_time_per_sample = inference_time / total if total > 0 else 0

        results[noise_level] = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'true_labels': all_true_labels
        }

        inference_times[noise_level] = {
            'total_inference_time': inference_time,
            'avg_time_per_sample': avg_inference_time_per_sample,
            'total_samples': total
        }

        print(f"噪声水平 {noise_level}: 准确率 = {accuracy:.2f}%")
        print(f"推理时间: {inference_time:.4f} 秒")
        print(f"平均每样本推理时间: {avg_inference_time_per_sample * 1000:.2f} 毫秒")

        # 计算并打印混淆矩阵
        cm = confusion_matrix(all_true_labels, all_predictions)
        print(f"混淆矩阵 (噪声水平 {noise_level}):")
        print(cm)

        # 打印分类报告
        print(f"\n分类报告 (噪声水平 {noise_level}):")
        print(classification_report(all_true_labels, all_predictions,
                                    target_names=['Carrageenan', 'Composite']))

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                    xticklabels=['Carrageenan', 'Composite'],
                    yticklabels=['Carrageenan', 'Composite'])
        plt.title(f'Confusion Matrix (Noise Level: {noise_level}, Accuracy: {accuracy:.2f}%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/carrageenan_vs_composite/confusion_matrix_noise_{noise_level}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 绘制噪声水平与准确率的关系图（包含推理时间）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # 准确率子图
    noise_levels_list = list(results.keys())
    accuracies = [results[level]['accuracy'] for level in noise_levels_list]

    ax1.plot(noise_levels_list, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level (Standard Deviation)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy vs. Noise Level')
    ax1.grid(True, alpha=0.3)

    # 标注每个点的准确率
    for i, (noise, acc) in enumerate(zip(noise_levels_list, accuracies)):
        ax1.annotate(f'{acc:.1f}%', (noise, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    # 推理时间子图
    inference_times_ms = [inference_times[level]['avg_time_per_sample'] * 1000 for level in noise_levels_list]

    ax2.plot(noise_levels_list, inference_times_ms, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level (Standard Deviation)')
    ax2.set_ylabel('Average Inference Time per Sample (ms)')
    ax2.set_title('Inference Time vs. Noise Level')
    ax2.grid(True, alpha=0.3)

    # 标注每个点的推理时间
    for i, (noise, time_ms) in enumerate(zip(noise_levels_list, inference_times_ms)):
        ax2.annotate(f'{time_ms:.1f}ms', (noise, time_ms), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig('results/carrageenan_vs_composite/accuracy_and_time_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存噪声测试结果
    noise_results_df = pd.DataFrame({
        'Noise_Level': noise_levels_list,
        'Accuracy': accuracies,
        'Avg_Inference_Time_ms': inference_times_ms
    })
    noise_results_df.to_csv('results/carrageenan_vs_composite/noise_test_results.csv', index=False)
    print("噪声测试结果已保存到 results/carrageenan_vs_composite/noise_test_results.csv")

    # ==================================================
    # 生成完整的时间报告
    # ==================================================
    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("完整时间性能报告")
    print("=" * 60)

    print(f"数据预处理阶段:")
    print(f"  - 伪图生成时间: {pseudo_time:.2f} 秒")
    print(f"  - 特征提取时间: {feature_extraction_time:.2f} 秒")
    print(f"  - 总预处理时间: {pseudo_time + feature_extraction_time:.2f} 秒")

    print(f"\n训练阶段:")
    print(f"  - 总训练时间: {total_training_time:.2f} 秒")
    print(f"  - 训练epoch数量: {len(epoch_times)}")
    print(f"  - 平均每个epoch时间: {np.mean(epoch_times):.2f} 秒")
    print(f"  - 最快epoch时间: {np.min(epoch_times):.2f} 秒")
    print(f"  - 最慢epoch时间: {np.max(epoch_times):.2f} 秒")

    print(f"\n推理阶段 (不同噪声水平):")
    for noise_level in noise_levels:
        time_info = inference_times[noise_level]
        acc_info = results[noise_level]
        print(f"  - 噪声 std={noise_level}:")
        print(f"     准确率: {acc_info['accuracy']:.2f}%")
        print(f"     总推理时间: {time_info['total_inference_time']:.4f} 秒")
        print(f"     平均每样本推理时间: {time_info['avg_time_per_sample'] * 1000:.2f} 毫秒")
        print(f"     推理速度: {1 / time_info['avg_time_per_sample']:.1f} 样本/秒")

    print(f"\n总体统计:")
    print(f"  - 程序总运行时间: {total_time:.2f} 秒")
    print(
        f"  - 训练+测试时间占比: {(total_training_time + sum(inference_times[nl]['total_inference_time'] for nl in noise_levels)) / total_time * 100:.1f}%")
    print(f"  - 数据预处理时间占比: {(pseudo_time + feature_extraction_time) / total_time * 100:.1f}%")

    # 打印详细结果
    print("\n噪声测试结果汇总:")
    print("Noise Level | Accuracy | Avg Inference Time")
    print("-" * 45)
    for noise_level in noise_levels:
        acc = results[noise_level]['accuracy']
        avg_time_ms = inference_times[noise_level]['avg_time_per_sample'] * 1000
        print(f"{noise_level:>10} | {acc:>8.2f}% | {avg_time_ms:>8.2f}ms")

    # 保存完整模型（包含时间性能信息）
    torch.save({
        'impedance_encoder': impedance_encoder.state_dict(),
        'cross_modal_module': cross_modal_module.state_dict(),
        'classifier': model.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'feature_scaler': feature_scaler,
        'spectral_scaler': spectral_scaler,
        'noise_test_results': results,
        'inference_times': inference_times,
        'training_time': total_training_time,
        'best_val_acc': best_val_acc,
        'performance_metrics': {
            'total_training_time': total_training_time,
            'avg_epoch_time': np.mean(epoch_times),
            'baseline_inference_time_per_sample': inference_times[0.0]['avg_time_per_sample'],
            'baseline_accuracy': results[0.0]['accuracy']
        }
    }, 'models/carrageenan_vs_composite/full_model.pth')
    print("\n完整模型已保存到 models/carrageenan_vs_composite/full_model.pth")