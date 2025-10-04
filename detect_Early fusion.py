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
import time  # 添加时间模块


# ==================================================
# 1. 谱数据处理与伪图生成（修改为每个样本生成伪图）
# ==================================================
def generate_pseudo_images_per_sample():
    # 读取Excel数据
    df = pd.read_excel('卡拉胶复合胶增强.xlsx')

    # 确保目录存在
    os.makedirs('pseudo_images/classification', exist_ok=True)

    # 定义频率和特征
    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    # 存储样本ID和类别映射
    sample_id_map = {}

    # 为每个样本生成伪图
    for idx, row in df.iterrows():
        sample_id = f"sample_{idx}"

        # 获取类型信息（卡拉胶=1，复合胶=2）
        sample_type = row['类型']
        class_name = "Carrageenan" if sample_type == 1 else "Composite"

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
        plt.title(f'Sample ID: {sample_id}, Type: {class_name}')
        plt.xlabel('Features')
        plt.ylabel('Frequency (Hz)')
        plt.xticks(range(len(features)), features)
        plt.yticks(range(len(frequencies)), [str(f) for f in frequencies])

        # 保存伪图（使用样本ID作为文件名）
        image_path = f'pseudo_images/classification/{sample_id}.png'
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 存储样本映射关系
        sample_id_map[sample_id] = {
            'class': class_name,
            'class_id': 0 if class_name == "Carrageenan" else 1,
            'image_path': image_path
        }

        print(f'Generated pseudo image: {image_path} - Type: {class_name}')

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
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================================================
# 4. 分类器模型
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
# 5. 保存数据集函数
# ==================================================
def save_dataset(dataset, file_path):
    """保存数据集到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save({
        'features': dataset.features,
        'labels': dataset.labels
    }, file_path)
    print(f"数据集已保存到: {file_path}")


# ==================================================
# 6. 绘制混淆矩阵
# ==================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    plt.show()

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return cm


# ==================================================
# 7. 添加高斯噪声的函数
# ==================================================
def add_gaussian_noise(features, mean=0, std=0.1):
    """
    向特征添加高斯噪声
    Args:
        features: 特征张量
        mean: 噪声均值
        std: 噪声标准差
    """
    noise = torch.randn_like(features) * std + mean
    noisy_features = features + noise
    return noisy_features


# ==================================================
# 8. 测试集加噪声评估函数（添加时间测量）
# ==================================================
def test_with_noise(model, test_loader, device, noise_std=0.1):
    """
    在测试集上添加高斯噪声并评估模型
    """
    model.eval()
    all_preds = []
    all_labels = []

    start_time = time.time()  # 开始计时

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            # 添加高斯噪声
            noisy_features = add_gaussian_noise(features, std=noise_std)

            # 使用加噪声后的特征进行预测
            outputs = model(noisy_features)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()  # 结束计时
    inference_time = end_time - start_time

    # 计算平均每个样本的推理时间
    total_samples = len(all_labels)
    avg_time_per_sample = inference_time / total_samples if total_samples > 0 else 0

    return all_labels, all_preds, inference_time, avg_time_per_sample


# ==================================================
# 9. 绘制噪声鲁棒性曲线（添加时间显示）
# ==================================================
def plot_noise_robustness(noise_levels, accuracies, inference_times=None, save_path=None):
    """
    绘制噪声鲁棒性曲线
    """
    if inference_times is None:
        # 单图模式：只显示准确率
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.title('Model Robustness to Gaussian Noise', fontsize=16)
        plt.xlabel('Noise Standard Deviation', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(noise_levels)
    else:
        # 双图模式：显示准确率和推理时间
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 准确率子图
        ax1.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Model Robustness to Gaussian Noise', fontsize=16)
        ax1.set_ylabel('Accuracy (%)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(noise_levels)

        # 推理时间子图（转换为毫秒）
        inference_times_ms = [t * 1000 for t in inference_times]
        ax2.plot(noise_levels, inference_times_ms, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Noise Standard Deviation', fontsize=14)
        ax2.set_ylabel('Inference Time per Sample (ms)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(noise_levels)

    # 添加数据标签
    for i, (noise, acc) in enumerate(zip(noise_levels, accuracies)):
        if inference_times is None:
            plt.annotate(f'{acc:.1f}%', (noise, acc),
                         textcoords="offset points", xytext=(0, 10), ha='center')
        else:
            ax1.annotate(f'{acc:.1f}%', (noise, acc),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            # 在时间图上也标注时间
            time_ms = inference_times[i] * 1000
            ax2.annotate(f'{time_ms:.1f}ms', (noise, time_ms),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"噪声鲁棒性曲线已保存到: {save_path}")
    plt.show()


# ==================================================
# 10. 主流程（添加完整时间检测）
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
    sample_id_map = generate_pseudo_images_per_sample()
    pseudo_time = time.time() - pseudo_start_time
    print(f"伪图生成完成，耗时: {pseudo_time:.2f} 秒")

    # 步骤2: 初始化特征提取器
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # 步骤3: 提取所有样本的特征向量
    print("正在提取特征向量...")
    feature_extraction_start = time.time()
    all_features = []
    all_labels = []

    # 创建样本ID到RGB图像路径的映射
    rgb_image_map = {}
    class_names = ["Carrageenan", "Composite"]

    # 扫描RGB图像目录
    image_base_dir = 'detect_data - zq'
    for class_dir in os.listdir(image_base_dir):
        class_path = os.path.join(image_base_dir, class_dir)
        if os.path.isdir(class_path):
            # 确定类别ID
            if class_dir == "Carrageenan":
                class_id = 0
            elif class_dir == "Composite hydrocolloid":
                class_id = 1
            else:
                continue  # 跳过其他目录

            # 遍历该类别下的所有图像
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_id = f"sample_{len(rgb_image_map)}"
                    rgb_image_map[sample_id] = {
                        'image_path': os.path.join(class_path, img_file),
                        'class_id': class_id,
                        'class_name': class_names[class_id]
                    }

    print(f"检测到的类别: {class_names}")
    print(f"总共找到 {len(rgb_image_map)} 张RGB图像")

    # 提取特征
    processed_count = 0
    for sample_id, rgb_info in rgb_image_map.items():
        # 检查是否有对应的伪图
        if sample_id not in sample_id_map:
            print(f"警告: 样本 {sample_id} 没有对应的伪图数据")
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

        # 拼接特征向量
        combined_features = np.concatenate((pseudo_features, rgb_features))

        # 使用RGB图像的类别标签
        label = rgb_info['class_id']

        all_features.append(combined_features)
        all_labels.append(label)
        processed_count += 1

        if processed_count % 50 == 0:
            print(f"已处理 {processed_count} 个样本...")

    feature_extraction_time = time.time() - feature_extraction_start
    print(f"特征提取完成，耗时: {feature_extraction_time:.2f} 秒")

    # 检查是否有样本
    if len(all_features) == 0:
        print("错误: 没有提取到任何样本特征!")
        exit(1)

    print(f"成功处理 {len(all_features)} 个样本")

    # 转换为PyTorch张量
    features_tensor = torch.tensor(np.array(all_features), dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 打印类别分布
    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    print("\n类别分布:")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {class_names[label]}: {count} 个样本")

    # 步骤4: 划分数据集 (60%训练, 40%测试)
    print("\n划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_tensor, labels_tensor, test_size=0.4, random_state=42, stratify=labels_tensor
    )

    # 创建数据集对象
    train_dataset = FeatureDataset(X_train, y_train)
    test_dataset = FeatureDataset(X_test, y_test)

    # 保存数据集
    save_dataset(train_dataset, 'datasets/classification/train_dataset.pth')
    save_dataset(test_dataset, 'datasets/classification/test_dataset.pth')

    # 保存类别信息
    os.makedirs('models/classification', exist_ok=True)
    joblib.dump(class_names, 'models/classification/class_names.pkl')
    print("类别名称已保存到 models/classification/class_names.pkl")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 步骤5: 初始化分类器
    num_classes = len(class_names)
    input_size = X_train.shape[1]
    print(f"输入特征维度: {input_size}, 类别数: {num_classes}")

    model = FeatureClassifier(input_size, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("\n开始训练模型...")
    training_start_time = time.time()
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []

    # 从训练集中划分验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_dataset_split = FeatureDataset(X_train_split, y_train_split)
    val_dataset_split = FeatureDataset(X_val_split, y_val_split)

    train_loader_split = DataLoader(train_dataset_split, batch_size=32, shuffle=True)
    val_loader_split = DataLoader(val_dataset_split, batch_size=32, shuffle=False)

    for epoch in range(50):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader_split:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader_split)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader_split:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_accs.append(val_acc)

        print(
            f'Epoch [{epoch + 1}/50], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/classification/best_feature_classifier.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\n训练完成，总训练时间: {training_time:.2f} 秒")
    print(f"平均每个epoch时间: {avg_epoch_time:.2f} 秒")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/classification/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 加载最佳模型
    model.load_state_dict(torch.load('models/classification/best_feature_classifier.pth'))
    model.eval()

    # ==================================================
    # 11. 扩展的噪声鲁棒性测试（添加时间测量）
    # ==================================================
    print("\n" + "=" * 60)
    print("扩展的噪声鲁棒性测试")
    print("=" * 60)

    # 定义更广泛的噪声水平
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    noise_accuracies = []
    noise_inference_times = []  # 存储每个噪声水平的推理时间
    noise_avg_times = []  # 存储每个噪声水平的平均推理时间

    print(f"{'噪声水平':<10} {'准确率':<10} {'准确率下降':<12} {'总推理时间':<12} {'平均时间/样本':<15}")
    print("-" * 70)

    # 无噪声基准测试
    baseline_start_time = time.time()
    baseline_labels, baseline_preds, baseline_inference_time, baseline_avg_time = test_with_noise(model, test_loader,
                                                                                                  device, noise_std=0.0)
    baseline_acc = 100. * np.mean(np.array(baseline_labels) == np.array(baseline_preds))
    noise_accuracies.append(baseline_acc)
    noise_inference_times.append(baseline_inference_time)
    noise_avg_times.append(baseline_avg_time)
    print(
        f"{'0.0':<10} {baseline_acc:.2f}%{'':<12} {baseline_inference_time:.4f}s{'':<8} {baseline_avg_time * 1000:.2f}ms")

    # 绘制无噪声混淆矩阵
    cm_path = 'models/classification/confusion_matrix_noise_0.0.png'
    plot_confusion_matrix(baseline_labels, baseline_preds, class_names,
                          cm_path, title=f'Confusion Matrix (No Noise) - Acc: {baseline_acc:.2f}%')

    # 测试不同噪声水平
    for noise_std in noise_levels[1:]:  # 跳过0.0
        test_labels, test_preds, inference_time, avg_time = test_with_noise(model, test_loader, device, noise_std)
        test_acc = 100. * np.mean(np.array(test_labels) == np.array(test_preds))
        noise_accuracies.append(test_acc)
        noise_inference_times.append(inference_time)
        noise_avg_times.append(avg_time)

        accuracy_drop = baseline_acc - test_acc
        print(
            f"{noise_std:<10} {test_acc:.2f}%    {accuracy_drop:.2f}%       {inference_time:.4f}s         {avg_time * 1000:.2f}ms")

        # 为关键噪声水平绘制混淆矩阵
        if noise_std in [0.1, 0.3, 0.5, 1.0]:
            cm_path = f'models/classification/confusion_matrix_noise_{noise_std}.png'
            plot_confusion_matrix(test_labels, test_preds, class_names,
                                  cm_path, title=f'Confusion Matrix (Noise std={noise_std}) - Acc: {test_acc:.2f}%')

    # 绘制噪声鲁棒性曲线（包含时间信息）
    robustness_path = 'models/classification/noise_robustness_curve.png'
    plot_noise_robustness(noise_levels, noise_accuracies, noise_avg_times, robustness_path)

    # 保存噪声测试结果（包含时间信息）
    noise_results = {
        'noise_levels': noise_levels,
        'accuracies': noise_accuracies,
        'inference_times': noise_inference_times,
        'avg_times_per_sample': noise_avg_times,
        'baseline_accuracy': baseline_acc
    }
    joblib.dump(noise_results, 'models/classification/noise_test_results.pkl')
    print(f"\n噪声测试结果已保存到: models/classification/noise_test_results.pkl")

    # 分析噪声鲁棒性
    print("\n" + "=" * 40)
    print("噪声鲁棒性分析")
    print("=" * 40)

    # 找到准确率下降到50%和25%的噪声水平
    fifty_percent_noise = None
    twenty_five_percent_noise = None

    for i, (noise, acc) in enumerate(zip(noise_levels, noise_accuracies)):
        if acc <= 50.0 and fifty_percent_noise is None:
            fifty_percent_noise = noise
        if acc <= 25.0 and twenty_five_percent_noise is None:
            twenty_five_percent_noise = noise

    print(f"基准准确率: {baseline_acc:.2f}%")
    if fifty_percent_noise:
        print(f"准确率下降到50%的噪声水平: std = {fifty_percent_noise}")
    else:
        print("在测试范围内准确率未下降到50%")

    if twenty_five_percent_noise:
        print(f"准确率下降到25%的噪声水平: std = {twenty_five_percent_noise}")
    else:
        print("在测试范围内准确率未下降到25%")

    # ==================================================
    # 12. 生成完整的时间报告
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
    print(f"  - 总训练时间: {training_time:.2f} 秒")
    print(f"  - 平均每个epoch时间: {avg_epoch_time:.2f} 秒")
    print(f"  - 最快epoch时间: {min(epoch_times):.2f} 秒")
    print(f"  - 最慢epoch时间: {max(epoch_times):.2f} 秒")

    print(f"\n推理阶段 (无噪声):")
    print(f"  - 总推理时间: {baseline_inference_time:.4f} 秒")
    print(f"  - 样本数量: {len(baseline_labels)}")
    print(f"  - 平均每样本推理时间: {baseline_avg_time * 1000:.2f} 毫秒")
    print(f"  - 推理速度: {1 / baseline_avg_time:.1f} 样本/秒")

    print(f"\n不同噪声水平下的推理性能:")
    for i, noise_std in enumerate([0.0, 0.1, 0.3, 0.5, 1.0, 2.0]):
        if i < len(noise_levels) and noise_std in noise_levels:
            idx = noise_levels.index(noise_std)
            print(f"  - 噪声 std={noise_std}: {noise_avg_times[idx] * 1000:.2f}ms/样本, "
                  f"准确率: {noise_accuracies[idx]:.2f}%")

    print(f"\n总体统计:")
    print(f"  - 程序总运行时间: {total_time:.2f} 秒")
    print(f"  - 训练+测试时间占比: {(training_time + baseline_inference_time) / total_time * 100:.1f}%")
    print(f"  - 数据预处理时间占比: {(pseudo_time + feature_extraction_time) / total_time * 100:.1f}%")

    # 保存完整模型（包含架构和时间性能信息）
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'class_names': class_names,
        'noise_robustness': noise_results,
        'performance_metrics': {
            'total_training_time': training_time,
            'avg_epoch_time': avg_epoch_time,
            'baseline_inference_time_per_sample': baseline_avg_time,
            'baseline_accuracy': baseline_acc,
            'best_validation_accuracy': best_val_acc
        }
    }, 'models/classification/full_model.pth')
    print("\n完整模型已保存到 models/classification/full_model.pth")

    print("\n训练和噪声测试完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集基准准确率: {baseline_acc:.2f}%")

    print(f"总程序运行时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
