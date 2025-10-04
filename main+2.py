import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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


# ==================================================
# 1. 谱数据处理与伪图生成（修改为每个样本生成伪图）
# ==================================================
def generate_pseudo_images_per_sample():
    # 读取Excel数据
    df = pd.read_excel('zq_spectra_data.xlsx')

    # 确保目录存在
    os.makedirs('pseudo_images/zq-full', exist_ok=True)

    # 定义频率和特征
    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    # 存储样本ID和浓度映射
    sample_id_map = {}
    # 存储原始谱数据（用于多频率阻抗编码器）
    spectral_data_map = {}

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

        # 存储原始谱数据（归一化前）
        spectral_data_map[sample_id] = feature_matrix.copy()

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
        image_path = f'pseudo_images/zq-full/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 存储样本映射关系
        sample_id_map[sample_id] = {
            'concentration': conc_percent,
            'image_path': image_path
        }

        print(f'Generated pseudo image: {image_path}')

    return sample_id_map, spectral_data_map


# ==================================================
# 2. 特征提取器
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
# 3. 多频率阻抗编码器（改进版）
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
# 4. 跨模态物理一致性模块（改进版）
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
# 5. 数据集类
# ==================================================
class FeatureDataset(Dataset):
    def __init__(self, features, labels, spectral_data=None):
        self.features = features
        self.labels = labels
        self.spectral_data = spectral_data  # 用于多频率阻抗编码器的原始谱数据

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.spectral_data is not None:
            return self.features[idx], self.labels[idx], self.spectral_data[idx]
        return self.features[idx], self.labels[idx]


# ==================================================
# 6. 分类器模型（改进版）
# ==================================================
class FeatureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
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
# 7. 保存数据集函数
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
# 8. 主流程（改进版）
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
    sample_id_map, spectral_data_map = generate_pseudo_images_per_sample()

    # 步骤2: 初始化特征提取器
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # 初始化多频率阻抗编码器
    impedance_encoder = MultiFrequencyImpedanceEncoder().to(device)

    # 初始化跨模态一致性模块
    cross_modal_module = CrossModalPhysicalConsistency().to(device)

    # 步骤3: 提取所有样本的特征向量
    print("正在提取特征向量...")
    all_features = []
    all_labels = []
    all_spectral_data = []  # 用于存储原始谱数据

    # 创建样本ID到RGB图像路径的映射
    rgb_image_map = {}
    concentrations = set()

    # 扫描RGB图像目录
    image_base_dir = 'zq_beef_images_resized'
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

        # 获取原始谱数据
        spectral_data = spectral_data_map[sample_id]

        # 拼接特征向量
        combined_features = np.concatenate((pseudo_features, rgb_features))
        # 使用标签编码器转换标签
        label = label_encoder.transform([rgb_info['concentration']])[0]

        all_features.append(combined_features)
        all_labels.append(label)
        all_spectral_data.append(spectral_data)

    # 检查是否有样本
    if len(all_features) == 0:
        print("错误: 没有提取到任何样本特征!")
        exit(1)

    # 数据验证
    print("验证数据完整性...")
    all_features_np = np.array(all_features)
    all_spectral_np = np.array(all_spectral_data)

    # 检查NaN和Inf
    if np.isnan(all_features_np).any() or np.isinf(all_features_np).any():
        print("警告: 特征数据包含NaN或Inf值!")
        # 替换为0
        all_features_np = np.nan_to_num(all_features_np, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(all_spectral_np).any() or np.isinf(all_spectral_np).any():
        print("警告: 谱数据包含NaN或Inf值!")
        # 替换为0
        all_spectral_np = np.nan_to_num(all_spectral_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 特征标准化
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(all_features_np)

    # 谱数据标准化
    spectral_scaler = StandardScaler()
    # 将谱数据从3D转换为2D用于标准化
    spectral_2d = all_spectral_np.reshape(all_spectral_np.shape[0], -1)
    scaled_spectral = spectral_scaler.fit_transform(spectral_2d)
    # 恢复为3D形状
    scaled_spectral = scaled_spectral.reshape(all_spectral_np.shape)

    # 转换为PyTorch张量
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    spectral_data_tensor = torch.tensor(scaled_spectral, dtype=torch.float32)

    # 打印标签分布
    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 个样本 (浓度: {label_encoder.inverse_transform([label])[0]}%)")

    # 计算类别权重（处理类别不平衡）
    class_weights = 1.0 / counts.float()
    class_weights = class_weights / class_weights.sum()
    print(f"类别权重: {class_weights}")

    # 步骤4: 划分数据集 (70%训练, 30%验证)
    print("划分数据集...")
    X_train, X_val, y_train, y_val, S_train, S_val = train_test_split(
        features_tensor, labels_tensor, spectral_data_tensor,
        test_size=0.3, random_state=42, stratify=labels_tensor
    )

    # 创建数据集对象（包含谱数据）
    train_dataset = FeatureDataset(X_train, y_train, S_train)
    val_dataset = FeatureDataset(X_val, y_val, S_val)

    # 保存数据集
    os.makedirs('datasets/zq-full', exist_ok=True)
    save_dataset(train_dataset, 'datasets/zq-full/train_dataset.pth')
    save_dataset(val_dataset, 'datasets/zq-full/val_dataset.pth')

    # 保存标签编码器
    os.makedirs('models/zq-full', exist_ok=True)
    joblib.dump(label_encoder, 'models/zq-full/label_encoder.pkl')
    print("标签编码器已保存到 models/zq-full/label_encoder.pkl")

    # 保存标签映射信息
    with open('datasets/zq-full/label_mapping.txt', 'w') as f:
        for original, encoded in label_mapping.items():
            f.write(f"{original}% -> {encoded}\n")
    print("标签映射信息已保存到 datasets/zq-full/label_mapping.txt")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小批量大小
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 步骤5: 初始化分类器
    num_classes = len(label_encoder.classes_)

    # 输入尺寸为融合特征尺寸 + 跨模态模块输出尺寸
    fusion_feat_dim = 512  # 跨模态模块输出尺寸
    input_size = X_train.shape[1] + fusion_feat_dim

    print(f"输入特征维度: {input_size}, 类别数: {num_classes}")

    model = FeatureClassifier(input_size, num_classes).to(device)

    # 损失函数和优化器（使用类别权重）
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
    consistency_weight = 0.1  # 降低一致性权重

    # 训练模型
    print("开始训练模型...")
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_patience = 10
    patience_counter = 0

    for epoch in range(50):  # 增加训练轮数
        model.train()
        impedance_encoder.train()
        cross_modal_module.train()

        train_loss = 0.0
        train_consistency_loss = 0.0
        train_class_loss = 0.0
        correct = 0
        total = 0

        # 添加梯度监控
        total_grad_norm = 0.0

        for features, labels, spectral_data in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            spectral_data = spectral_data.to(device)

            optimizer.zero_grad()

            # 1. 多频率阻抗编码
            impedance_features, _ = impedance_encoder(spectral_data)

            # 2. 跨模态一致性处理
            # 拆分特征：前2048维是伪图特征，后2048维是RGB特征
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

            # 5. 总损失 = 分类损失 + 一致性损失
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

        # 计算平均损失和准确率
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        train_consistency_loss = train_consistency_loss / len(train_loader)
        train_class_loss = train_class_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)

        # 验证
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

                # 计算验证损失
                loss = criterion(outputs, labels) + consistency_weight * consistency_loss
                val_loss += loss.item()

                # 4. 计算准确率
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
            }, 'models/zq-full/best_feature_classifier.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停: 验证准确率连续 {early_stop_patience} 个epoch未提升")
                break

    # 加载最佳模型并做最终评估
    print(f"加载最佳模型 (来自epoch {best_epoch + 1}, 验证准确率: {best_val_acc:.2f}%)")
    checkpoint = torch.load('models/zq-full/best_feature_classifier.pth')
    impedance_encoder.load_state_dict(checkpoint['impedance_encoder'])
    cross_modal_module.load_state_dict(checkpoint['cross_modal_module'])
    model.load_state_dict(checkpoint['classifier'])

    impedance_encoder.eval()
    cross_modal_module.eval()
    model.eval()

    final_correct = 0
    final_total = 0

    with torch.no_grad():
        for features, labels, spectral_data in val_loader:
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
            final_total += labels.size(0)
            final_correct += predicted.eq(labels).sum().item()

    final_acc = 100. * final_correct / final_total
    print(f"最终验证准确率: {final_acc:.2f}%")

    # 保存完整模型
    torch.save({
        'impedance_encoder': impedance_encoder.state_dict(),
        'cross_modal_module': cross_modal_module.state_dict(),
        'classifier': model.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'feature_scaler': feature_scaler,
        'spectral_scaler': spectral_scaler,
        'val_acc': final_acc
    }, 'models/zq-full/full_model.pth')

    print("完整模型已保存到 models/zq-full/full_model.pth")
