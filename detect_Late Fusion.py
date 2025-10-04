import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
import time
import datetime


# ==================================================
# 1. 谱数据处理与伪图生成（修改为每个样本生成伪图）
# ==================================================
def generate_pseudo_images_per_sample():
    # 读取Excel数据
    df = pd.read_excel('D:/开题/电谱/BeefClassifier/data/卡拉胶复合胶增强.xlsx')

    # 确保目录存在
    os.makedirs('pseudo_images/carrageenan_composite', exist_ok=True)

    # 定义频率和特征
    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    # 存储样本ID和类型映射
    sample_id_map = {}

    # 为每个样本生成伪图
    for idx, row in df.iterrows():
        sample_id = f"sample_{idx}"

        # 获取类型信息（假设第一列是类型，1=卡拉胶，2=复合胶）
        sample_type = int(row.iloc[0])  # 获取第一列的值
        type_name = "Carrageenan" if sample_type == 1 else "Composite"

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
        image_path = f'pseudo_images/carrageenan_composite/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 存储样本映射关系
        sample_id_map[sample_id] = {
            'type': type_name,
            'type_code': 0 if type_name == "Carrageenan" else 1,  # 卡拉胶=0, 复合胶=1
            'image_path': image_path
        }

        print(f'Generated pseudo image: {image_path} - Type: {type_name}')

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
# 4. Late Fusion 分类器模型（修改为二分类）
# ==================================================
class LateFusionClassifier(nn.Module):
    def __init__(self, pseudo_feature_size, rgb_feature_size, num_classes=2):
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

        # 融合分类器（二分类）
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
# 6. 绘制混淆矩阵函数
# ==================================================
def plot_confusion_matrix(true_labels, predictions, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印分类报告
    print(f"\n{title} Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))


# ==================================================
# 7. 添加高斯噪声的函数
# ==================================================
def add_gaussian_noise(tensor, mean=0., std=0.1):
    """为张量添加高斯噪声"""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise


# ==================================================
# 8. 测试集加噪评估函数（添加时间测量）
# ==================================================
def evaluate_with_gaussian_noise(model, test_loader, device, noise_std=0.1):
    """在测试集上加高斯噪声并评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\n正在评估加噪测试集 (噪声标准差: {noise_std})...")

    start_time = time.time()

    with torch.no_grad():
        for pseudo_features, rgb_features, labels in test_loader:
            # 为特征添加高斯噪声
            pseudo_features_noisy = add_gaussian_noise(pseudo_features, std=noise_std)
            rgb_features_noisy = add_gaussian_noise(rgb_features, std=noise_std)

            pseudo_features_noisy = pseudo_features_noisy.to(device)
            rgb_features_noisy = rgb_features_noisy.to(device)
            labels = labels.to(device)

            outputs = model(pseudo_features_noisy, rgb_features_noisy)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time

    # 计算每个样本的平均推理时间
    total_samples = len(all_labels)
    avg_time_per_sample = inference_time / total_samples

    return all_labels, all_preds, inference_time, avg_time_per_sample


# ==================================================
# 9. 扩展的多噪声水平评估函数（添加时间测量）
# ==================================================
def evaluate_extended_noise_levels(model, test_loader, device, class_names):
    """在更广泛的噪声水平下评估模型"""
    # 扩展的噪声水平：从轻微到极端
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = {}
    timing_results = {}

    print("开始扩展噪声水平评估...")
    print(f"测试的噪声水平: {noise_levels}")

    for noise_std in noise_levels:
        if noise_std == 0.0:
            # 无噪声情况
            model.eval()
            all_preds = []
            all_labels = []

            start_time = time.time()

            with torch.no_grad():
                for pseudo_features, rgb_features, labels in test_loader:
                    pseudo_features = pseudo_features.to(device)
                    rgb_features = rgb_features.to(device)
                    labels = labels.to(device)

                    outputs = model(pseudo_features, rgb_features)
                    _, predicted = outputs.max(1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            end_time = time.time()
            inference_time = end_time - start_time
            total_samples = len(all_labels)
            avg_time_per_sample = inference_time / total_samples

        else:
            # 有噪声情况
            all_labels, all_preds, inference_time, avg_time_per_sample = evaluate_with_gaussian_noise(
                model, test_loader, device, noise_std)

        # 计算准确率
        accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        results[noise_std] = {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }

        timing_results[noise_std] = {
            'total_inference_time': inference_time,
            'avg_time_per_sample': avg_time_per_sample,
            'total_samples': len(all_labels)
        }

        print(f"噪声标准差 {noise_std}: 准确率 = {accuracy:.2f}%, "
              f"推理时间 = {inference_time:.4f}s, "
              f"平均每样本 = {avg_time_per_sample * 1000:.2f}ms")

        # 为关键噪声水平绘制混淆矩阵
        if noise_std in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            plot_confusion_matrix(all_labels, all_preds, class_names,
                                  f'Confusion Matrix (Noise std={noise_std})')

    return results, timing_results


# ==================================================
# 10. 绘制扩展的噪声鲁棒性曲线
# ==================================================
def plot_extended_noise_robustness(results, timing_results, save_path='results/extended_noise_robustness_curve.png'):
    """绘制模型在更广泛噪声水平下的准确率曲线"""
    noise_levels = sorted(results.keys())
    accuracies = [results[noise_std]['accuracy'] for noise_std in noise_levels]
    inference_times = [timing_results[noise_std]['avg_time_per_sample'] * 1000 for noise_std in noise_levels]  # 转换为毫秒

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：准确率曲线
    ax1.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Noise Standard Deviation')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Extended Model Robustness to Gaussian Noise')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(noise_levels)

    # 标注每个点的准确率
    for i, (noise, acc) in enumerate(zip(noise_levels, accuracies)):
        ax1.annotate(f'{acc:.1f}%', (noise, acc), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)

    # 子图2：推理时间曲线
    ax2.plot(noise_levels, inference_times, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Noise Standard Deviation')
    ax2.set_ylabel('Inference Time per Sample (ms)')
    ax2.set_title('Inference Time vs Noise Level')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(noise_levels)

    # 标注每个点的推理时间
    for i, (noise, time_ms) in enumerate(zip(noise_levels, inference_times)):
        ax2.annotate(f'{time_ms:.1f}ms', (noise, time_ms), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"扩展噪声鲁棒性曲线已保存到: {save_path}")


# ==================================================
# 11. 噪声鲁棒性分析函数
# ==================================================
def analyze_noise_robustness(results, timing_results):
    """分析模型的噪声鲁棒性"""
    baseline_acc = results[0.0]['accuracy']
    baseline_time = timing_results[0.0]['avg_time_per_sample'] * 1000  # 转换为毫秒

    print("\n" + "=" * 60)
    print("噪声鲁棒性分析报告")
    print("=" * 60)
    print(f"基准准确率 (无噪声): {baseline_acc:.2f}%")
    print(f"基准推理时间 (无噪声): {baseline_time:.2f}ms/样本")

    # 分析不同噪声水平下的性能下降
    critical_noise_levels = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    print("\n关键噪声水平下的性能:")
    for noise_std in critical_noise_levels:
        if noise_std in results:
            acc = results[noise_std]['accuracy']
            drop = baseline_acc - acc
            drop_percentage = (drop / baseline_acc) * 100

            time_ms = timing_results[noise_std]['avg_time_per_sample'] * 1000
            time_change = time_ms - baseline_time
            time_change_percentage = (time_change / baseline_time) * 100

            print(f"噪声 std={noise_std}: "
                  f"准确率={acc:.2f}%, 下降={drop:.2f}% ({drop_percentage:.1f}%) | "
                  f"时间={time_ms:.2f}ms, 变化={time_change:+.2f}ms ({time_change_percentage:+.1f}%)")

    # 找到准确率下降到特定阈值的噪声水平
    thresholds = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    threshold_noise = {}

    for threshold in thresholds:
        for noise_std in sorted(results.keys()):
            if results[noise_std]['accuracy'] <= threshold:
                threshold_noise[threshold] = {
                    'noise_std': noise_std,
                    'inference_time': timing_results[noise_std]['avg_time_per_sample'] * 1000
                }
                break

    print(f"\n准确率下降到特定阈值的噪声水平:")
    for threshold, data in threshold_noise.items():
        print(f"下降到 {threshold}% 以下: 噪声 std ≥ {data['noise_std']}, "
              f"推理时间 = {data['inference_time']:.2f}ms")

    return threshold_noise


# ==================================================
# 12. 时间测量装饰器
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
# 13. 主流程
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
    start_time = time.time()
    sample_id_map = generate_pseudo_images_per_sample()
    pseudo_image_time = time.time() - start_time
    print(f"伪图生成完成，耗时: {pseudo_image_time:.2f} 秒")

    # 步骤2: 初始化特征提取器
    pseudo_extractor = FeatureExtractor().to(device)
    rgb_extractor = FeatureExtractor().to(device)
    pseudo_extractor.eval()
    rgb_extractor.eval()

    # 步骤3: 提取所有样本的特征向量
    print("正在提取特征向量...")
    start_time = time.time()
    all_pseudo_features = []
    all_rgb_features = []
    all_labels = []

    # 创建样本ID到RGB图像路径的映射
    rgb_image_map = {}

    # 扫描RGB图像目录（detect data文件夹）
    image_base_dir = 'D:/开题/电谱/BeefClassifier/data/detect_data - zq'
    type_folders = ['Carrageenan', 'Composite hydrocolloid']
    type_mapping = {'Carrageenan': 0, 'Composite hydrocolloid': 1}  # 卡拉胶=0, 复合胶=1

    sample_counter = 0

    for type_folder in type_folders:
        type_path = os.path.join(image_base_dir, type_folder)
        if not os.path.exists(type_path):
            print(f"警告: 目录不存在: {type_path}")
            continue

        type_code = type_mapping[type_folder]
        type_name = "Carrageenan" if type_code == 0 else "Composite"

        # 获取该类型下的所有图像文件
        image_files = [f for f in os.listdir(type_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"在 {type_folder} 中找到 {len(image_files)} 张图像")

        for img_file in image_files:
            sample_id = f"sample_{sample_counter}"
            rgb_image_map[sample_id] = {
                'image_path': os.path.join(type_path, img_file),
                'type': type_name,
                'type_code': type_code
            }
            sample_counter += 1

    print(f"总共找到 {len(rgb_image_map)} 张RGB图像")

    # 确保有足够的样本
    if len(rgb_image_map) == 0:
        print("错误: 没有找到任何RGB图像!")
        exit(1)

    # 提取特征
    feature_extraction_start = time.time()
    for sample_id, rgb_info in rgb_image_map.items():
        # 提取伪图特征（如果存在对应的伪图）
        pseudo_path = sample_id_map.get(sample_id, {}).get('image_path', '')
        if os.path.exists(pseudo_path):
            try:
                pseudo_img = Image.open(pseudo_path).convert('RGB')
                pseudo_tensor = transform(pseudo_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    pseudo_features = pseudo_extractor(pseudo_tensor).cpu().numpy()
            except Exception as e:
                print(f"处理伪图 {pseudo_path} 时出错: {e}")
                # 如果伪图处理失败，使用零向量代替
                pseudo_features = np.zeros(2048)  # ResNet50特征维度
        else:
            print(f"警告: 伪图文件不存在: {pseudo_path}，使用零向量")
            pseudo_features = np.zeros(2048)  # ResNet50特征维度

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

        # 使用类型作为标签（卡拉胶=0, 复合胶=1）
        label = rgb_info['type_code']

        all_pseudo_features.append(pseudo_features)
        all_rgb_features.append(rgb_features)
        all_labels.append(label)

    feature_extraction_time = time.time() - feature_extraction_start
    print(f"特征提取完成，耗时: {feature_extraction_time:.2f} 秒")

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
        type_name = "Carrageenan" if label == 0 else "Composite"
        print(f"标签 {label} ({type_name}): {count} 个样本")

    # 步骤4: 划分数据集 (70%训练, 30%验证)
    print("划分数据集...")
    indices = list(range(len(labels_tensor)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.4, random_state=25, stratify=labels_tensor.numpy()
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
    save_dataset(train_dataset, 'datasets/carrageenan_composite/train_dataset.pth')
    save_dataset(val_dataset, 'datasets/carrageenan_composite/val_dataset.pth')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 步骤5: 初始化Late Fusion分类器（二分类）
    num_classes = 2  # 卡拉胶 vs 复合胶
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

    # 用于存储训练历史
    train_losses = []
    train_accs = []
    val_accs = []

    # 训练时间测量
    total_training_time = 0

    for epoch in range(50):
        epoch_start_time = time.time()
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

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for pseudo_features, rgb_features, labels in val_loader:
                pseudo_features = pseudo_features.to(device)
                rgb_features = rgb_features.to(device)
                labels = labels.to(device)

                outputs = model(pseudo_features, rgb_features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_acc = 100. * val_correct / val_total
        val_accs.append(val_acc)

        print(
            f'Epoch [{epoch + 1}/50], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, 时间: {epoch_time:.2f}s')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models/carrageenan_composite', exist_ok=True)
            torch.save(model.state_dict(), 'models/carrageenan_composite/best_late_fusion_classifier.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    print(f"\n总训练时间: {total_training_time:.2f} 秒")
    print(f"平均每个epoch训练时间: {total_training_time / 50:.2f} 秒")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 加载最佳模型并做最终评估
    model.load_state_dict(torch.load('models/carrageenan_composite/best_late_fusion_classifier.pth'))
    model.eval()

    # 最终验证集评估
    final_correct = 0
    final_total = 0
    all_final_preds = []
    all_final_labels = []

    final_eval_start = time.time()

    with torch.no_grad():
        for pseudo_features, rgb_features, labels in val_loader:
            pseudo_features = pseudo_features.to(device)
            rgb_features = rgb_features.to(device)
            labels = labels.to(device)

            outputs = model(pseudo_features, rgb_features)
            _, predicted = outputs.max(1)
            final_total += labels.size(0)
            final_correct += predicted.eq(labels).sum().item()

            all_final_preds.extend(predicted.cpu().numpy())
            all_final_labels.extend(labels.cpu().numpy())

    final_eval_time = time.time() - final_eval_start
    final_acc = 100. * final_correct / final_total
    print(f"最终验证准确率: {final_acc:.2f}%")
    print(f"最终验证推理时间: {final_eval_time:.2f} 秒")
    print(f"平均每样本推理时间: {final_eval_time / len(all_final_labels) * 1000:.2f} 毫秒")

    # 绘制混淆矩阵
    class_names = ['Carrageenan', 'Composite']
    plot_confusion_matrix(all_final_labels, all_final_preds, class_names, 'Final Validation Confusion Matrix')

    # ==================================================
    # 新增：扩展的测试集加高斯噪声评估
    # ==================================================
    print("\n" + "=" * 70)
    print("开始扩展的加噪测试集评估")
    print("=" * 70)

    # 在更广泛的噪声水平下评估模型
    extended_noise_results, timing_results = evaluate_extended_noise_levels(model, val_loader, device, class_names)

    # 绘制扩展的噪声鲁棒性曲线
    plot_extended_noise_robustness(extended_noise_results, timing_results)

    # 分析噪声鲁棒性
    threshold_noise = analyze_noise_robustness(extended_noise_results, timing_results)

    # 保存噪声评估结果
    os.makedirs('results/noise_evaluation', exist_ok=True)
    noise_results_path = 'results/noise_evaluation/extended_noise_evaluation_results.pkl'
    joblib.dump({
        'noise_results': extended_noise_results,
        'timing_results': timing_results,
        'threshold_analysis': threshold_noise
    }, noise_results_path)
    print(f"扩展噪声评估结果已保存到: {noise_results_path}")

    # ==================================================
    # 生成完整的时间报告
    # ==================================================
    print("\n" + "=" * 70)
    print("完整时间性能报告")
    print("=" * 70)

    # 计算各个阶段的时间
    total_samples = len(all_final_labels)

    print(f"数据预处理阶段:")
    print(f"  - 伪图生成时间: {pseudo_image_time:.2f}s")
    print(f"  - 特征提取时间: {feature_extraction_time:.2f}s")
    print(f"  - 总预处理时间: {pseudo_image_time + feature_extraction_time:.2f}s")

    print(f"\n训练阶段:")
    print(f"  - 总训练时间: {total_training_time:.2f}s")
    print(f"  - 平均每个epoch时间: {total_training_time / 50:.2f}s")

    print(f"\n推理阶段 (无噪声):")
    print(f"  - 总推理时间: {final_eval_time:.2f}s")
    print(f"  - 样本数量: {total_samples}")
    print(f"  - 平均每样本推理时间: {final_eval_time / total_samples * 1000:.2f}ms")

    print(f"\n不同噪声水平下的推理时间:")
    for noise_std in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        if noise_std in timing_results:
            time_info = timing_results[noise_std]
            print(f"  - 噪声 std={noise_std}: {time_info['avg_time_per_sample'] * 1000:.2f}ms/样本")

    # 保存完整模型（包含架构）
    torch.save({
        'model_state_dict': model.state_dict(),
        'pseudo_feature_size': pseudo_feature_size,
        'rgb_feature_size': rgb_feature_size,
        'num_classes': num_classes,
        'class_names': class_names,
        'noise_robustness_analysis': threshold_noise,
        'performance_metrics': {
            'training_time': total_training_time,
            'inference_time_per_sample': final_eval_time / total_samples,
            'baseline_accuracy': final_acc
        }
    }, 'models/carrageenan_composite/full_late_fusion_model.pth')
    print("\n完整模型已保存到 models/carrageenan_composite/full_late_fusion_model.pth")

    # 保存类型映射信息
    os.makedirs('datasets/carrageenan_composite', exist_ok=True)
    with open('datasets/carrageenan_composite/type_mapping.txt', 'w') as f:
        f.write("0: Carrageenan\n")
        f.write("1: Composite hydrocolloid\n")
    print("类型映射信息已保存到 datasets/carrageenan_composite/type_mapping.txt")

    print("\n训练和扩展噪声测试完成！")