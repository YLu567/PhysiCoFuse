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
# 1. 谱数据处理与伪图生成（修改为每个样本生成伪图）
# ==================================================
def generate_pseudo_images_per_sample():
    # 读取Excel数据
    df = pd.read_excel('D:/谱/BeefClassifier/data/zq_spectra_data.xlsx')

    # 确保目录存在
    os.makedirs('pseudo_images/zq', exist_ok=True)

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
        image_path = f'pseudo_images/zq/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 存储样本映射关系
        sample_id_map[sample_id] = {
            'concentration': conc_percent,
            'image_path': image_path
        }

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
# 3. 数据集类（修改为使用特征向量）
# ==================================================
class FeatureDataset(Dataset):
    def __init__(self, pseudo_features, rgb_features, labels):
        self.pseudo_features = pseudo_features
        self.rgb_features = rgb_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pseudo_features[idx], self.rgb_features[idx], self.labels[idx]


# ==================================================
# 4. Late Fusion 分类器模型
# ==================================================
class LateFusionClassifier(nn.Module):
    def __init__(self, pseudo_feature_size, rgb_feature_size, num_classes):
        super().__init__()

        # 伪图特征处理分支
        self.pseudo_branch = nn.Sequential(
            nn.Linear(pseudo_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # RGB图像特征处理分支
        self.rgb_branch = nn.Sequential(
            nn.Linear(rgb_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pseudo_x, rgb_x):
        # 分别处理两种特征
        pseudo_features = self.pseudo_branch(pseudo_x)
        rgb_features = self.rgb_branch(rgb_x)

        # 融合特征
        fused_features = torch.cat((pseudo_features, rgb_features), dim=1)

        # 分类
        output = self.fusion_classifier(fused_features)
        return output


# ==================================================
# 5. 保存数据集函数
# ==================================================
def save_dataset(dataset, file_path):
    """保存数据集到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save({
        'pseudo_features': dataset.pseudo_features,
        'rgb_features': dataset.rgb_features,
        'labels': dataset.labels
    }, file_path)
    print(f"数据集已保存到: {file_path}")


# ==================================================
# 6. 主流程
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
    pseudo_extractor = FeatureExtractor().to(device)
    rgb_extractor = FeatureExtractor().to(device)
    pseudo_extractor.eval()
    rgb_extractor.eval()

    # 步骤3: 提取所有样本的特征向量
    print("正在提取特征向量...")
    all_pseudo_features = []
    all_rgb_features = []
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
                pseudo_features = pseudo_extractor(pseudo_tensor).cpu().numpy()
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
                rgb_features = rgb_extractor(rgb_tensor).cpu().numpy()
        except Exception as e:
            print(f"处理RGB图像 {rgb_path} 时出错: {e}")
            continue

        # 使用标签编码器转换标签
        label = label_encoder.transform([rgb_info['concentration']])[0]

        all_pseudo_features.append(pseudo_features)
        all_rgb_features.append(rgb_features)
        all_labels.append(label)

    # 检查是否有样本
    if len(all_pseudo_features) == 0:
        print("错误: 没有提取到任何样本特征!")
        exit(1)

    # 转换为PyTorch张量
    pseudo_features_tensor = torch.tensor(np.array(all_pseudo_features), dtype=torch.float32)
    rgb_features_tensor = torch.tensor(np.array(all_rgb_features), dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 打印标签分布
    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 个样本 (浓度: {label_encoder.inverse_transform([label])[0]}%)")

    # 步骤4: 划分数据集 (70%训练, 30%验证)
    print("划分数据集...")
    indices = list(range(len(labels_tensor)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels_tensor.numpy()
    )

    # 创建数据集对象
    train_dataset = FeatureDataset(
        pseudo_features_tensor[train_indices],
        rgb_features_tensor[train_indices],
        labels_tensor[train_indices]
    )

    val_dataset = FeatureDataset(
        pseudo_features_tensor[val_indices],
        rgb_features_tensor[val_indices],
        labels_tensor[val_indices]
    )

    # 保存数据集
    save_dataset(train_dataset, 'datasets/zq/train_dataset.pth')
    save_dataset(val_dataset, 'datasets/zq/val_dataset.pth')

    # 保存标签编码器
    os.makedirs('models/zq', exist_ok=True)
    joblib.dump(label_encoder, 'models/zq/label_encoder.pkl')
    print("标签编码器已保存到 models/zq/label_encoder.pkl")

    # 保存标签映射信息
    os.makedirs('datasets/zq', exist_ok=True)
    with open('datasets/zq/label_mapping.txt', 'w') as f:
        for original, encoded in label_mapping.items():
            f.write(f"{original}% -> {encoded}\n")
    print("标签映射信息已保存到 datasets/zq/label_mapping.txt")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 步骤5: 初始化Late Fusion分类器
    num_classes = len(label_encoder.classes_)
    pseudo_feature_size = pseudo_features_tensor.shape[1]
    rgb_feature_size = rgb_features_tensor.shape[1]

    print(f"伪图特征维度: {pseudo_feature_size}, RGB特征维度: {rgb_feature_size}, 类别数: {num_classes}")

    model = LateFusionClassifier(pseudo_feature_size, rgb_feature_size, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("开始训练模型...")
    best_val_acc = 0.0

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for pseudo_features, rgb_features, labels in train_loader:
            pseudo_features = pseudo_features.to(device)
            rgb_features = rgb_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pseudo_features, rgb_features)
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
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for pseudo_features, rgb_features, labels in val_loader:
                pseudo_features = pseudo_features.to(device)
                rgb_features = rgb_features.to(device)
                labels = labels.to(device)

                outputs = model(pseudo_features, rgb_features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(
            f'Epoch [{epoch + 1}/50], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models/zq', exist_ok=True)
            torch.save(model.state_dict(), 'models/zq/best_late_fusion_classifier.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    # 加载最佳模型并做最终评估
    model.load_state_dict(torch.load('models/zq/best_late_fusion_classifier.pth'))
    model.eval()

    final_correct = 0
    final_total = 0

    with torch.no_grad():
        for pseudo_features, rgb_features, labels in val_loader:
            pseudo_features = pseudo_features.to(device)
            rgb_features = rgb_features.to(device)
            labels = labels.to(device)

            outputs = model(pseudo_features, rgb_features)
            _, predicted = outputs.max(1)
            final_total += labels.size(0)
            final_correct += predicted.eq(labels).sum().item()

    final_acc = 100. * final_correct / final_total
    print(f"最终验证准确率: {final_acc:.2f}%")

    # 保存完整模型（包含架构）
    torch.save({
        'model_state_dict': model.state_dict(),
        'pseudo_feature_size': pseudo_feature_size,
        'rgb_feature_size': rgb_feature_size,
        'num_classes': num_classes
    }, 'models/zq/full_late_fusion_model.pth')
    print("完整模型已保存到 models/zq/full_late_fusion_model.pth")