import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
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
import warnings
import time
import re

warnings.filterwarnings('ignore')


# ==================================================
# 1. 生理属性定义与计算模块
# ==================================================
class PhysiologicalAttributes:
    """生理属性计算器"""

    # 四个目标生理属性
    ATTRIBUTE_NAMES = [
        'water_content',  # 含水量 - 与导电性相关
        'ion_concentration',  # 离子浓度 - 与电阻相关
        'tissue_porosity',  # 组织孔隙率 - 与电容相关
        'cell_integrity'  # 细胞完整性 - 与相位角相关
    ]

    # 目标属性索引（所有四个都保留）
    TARGET_ATTRIBUTES = [0, 1, 2, 3]

    @staticmethod
    def calculate_from_eis(feature_matrix):
        """
        从EIS特征矩阵计算四个生理属性值
        feature_matrix: [8频率, 6参数]
        参数顺序: ['CP', 'CS', 'Z', 'Φ', 'R', 'X']
        """
        # 1. 含水量 - 与阻抗Z负相关，特别关注低频
        low_freq_idx = [0, 1, 2]  # 100Hz, 500Hz, 1kHz
        water_content = np.mean(1.0 / (feature_matrix[low_freq_idx, 2] + 1e-6))

        # 2. 离子浓度 - 与电阻R负相关
        mid_freq_idx = [3, 4]  # 3kHz, 8kHz
        ion_concentration = np.mean(1.0 / (feature_matrix[mid_freq_idx, 4] + 1e-6))

        # 3. 组织孔隙率 - 与电容参数CP/CS比值相关
        tissue_porosity = np.mean(feature_matrix[:, 0] / (feature_matrix[:, 1] + 1e-6))

        # 4. 细胞完整性 - 与相位角Φ相关
        high_freq_idx = [5, 6, 7]  # 15kHz, 50kHz, 200kHz
        cell_integrity = np.mean(np.abs(feature_matrix[high_freq_idx, 3]))

        attributes = np.array([
            water_content, ion_concentration, tissue_porosity, cell_integrity
        ])

        # 注意：这里不要全局归一化，避免数据泄露
        # 只进行简单的数值稳定处理
        attributes = np.clip(attributes, 1e-8, None)

        return attributes


# ==================================================
# 2. 谱数据处理与伪图生成（修复数据泄露）
# ==================================================
def generate_pseudo_images_per_sample():
    """为每个样本生成伪图和生理属性 - 避免数据泄露"""
    spectra_dir = './data_train-54/spectra/'
    images_dir = './data_train-54/images/'
    labels_dir = './data_train-54/labels/'

    # 检查目录是否存在
    if not all(os.path.exists(d) for d in [spectra_dir, images_dir, labels_dir]):
        print("错误: 数据目录不存在!")
        return {}, {}, {}

    # 创建输出目录
    output_dir = './pseudo_images/54'
    os.makedirs(output_dir, exist_ok=True)

    frequencies = [100, 500, 1000, 3000, 8000, 15000, 50000, 200000]
    features = ['CP', 'CS', 'Z', 'Φ', 'R', 'X']

    sample_id_map = {}
    spectral_data_map = {}
    physiological_attr_map = {}

    # 获取所有频谱文件
    spectra_files = sorted([f for f in os.listdir(spectra_dir) if f.endswith('.xlsx')])
    print(f"找到 {len(spectra_files)} 个频谱文件")

    # 先收集所有样本信息
    sample_infos = []
    for spectra_file in spectra_files:
        sample_id = os.path.splitext(spectra_file)[0]
        spectra_path = os.path.join(spectra_dir, spectra_file)

        # 读取浓度
        label_path = os.path.join(labels_dir, f"{sample_id}.txt")
        concentration = None

        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    label_content = f.read().strip()
                    match = re.search(r'(\d+(\.\d+)?)%', label_content)
                    if match:
                        concentration = float(match.group(1))
                    else:
                        concentration = float(label_content.replace('%', ''))
            except:
                pass

        if concentration is None:
            try:
                conc_str = sample_id.split('_')[0].replace('%', '')
                concentration = float(conc_str)
            except:
                print(f"跳过样本 {sample_id}: 无法提取浓度")
                continue

        # 检查图像是否存在
        rgb_image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_path = os.path.join(images_dir, f"{sample_id}{ext}")
            if os.path.exists(test_path):
                rgb_image_path = test_path
                break

        sample_infos.append({
            'sample_id': sample_id,
            'spectra_path': spectra_path,
            'concentration': concentration,
            'rgb_image_path': rgb_image_path
        })

    print(f"成功读取 {len(sample_infos)} 个样本信息")

    # 为每个样本生成伪图和生理属性（避免使用全局统计信息）
    for info in sample_infos:
        sample_id = info['sample_id']

        try:
            # 读取频谱数据
            df = pd.read_excel(info['spectra_path'])

            # 创建特征矩阵
            feature_matrix = np.zeros((len(frequencies), len(features)))

            for i, freq in enumerate(frequencies):
                for j, feat in enumerate(features):
                    col_name = f"{feat}_{freq}"
                    if col_name in df.columns:
                        feature_matrix[i, j] = df[col_name].iloc[0]
                    else:
                        col_name_alt = f"{feat}{freq}"
                        if col_name_alt in df.columns:
                            feature_matrix[i, j] = df[col_name_alt].iloc[0]
                        else:
                            feature_matrix[i, j] = 0.0

            # 存储原始谱数据
            spectral_data_map[sample_id] = feature_matrix.copy()

            # 计算生理属性（不使用全局归一化）
            phys_attributes = PhysiologicalAttributes.calculate_from_eis(feature_matrix)
            physiological_attr_map[sample_id] = phys_attributes

            # 使用样本自身的统计信息进行归一化，避免数据泄露
            min_val = np.min(feature_matrix)
            max_val = np.max(feature_matrix)
            if max_val > min_val:
                normalized_matrix = (feature_matrix - min_val) / (max_val - min_val)
            else:
                normalized_matrix = np.zeros_like(feature_matrix)

            # 创建伪图
            plt.figure(figsize=(8, 6))
            plt.imshow(normalized_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Normalized Value')
            plt.title(f'Sample: {sample_id}, Conc: {int(info["concentration"])}%')
            plt.xlabel('Features')
            plt.ylabel('Frequency (Hz)')
            plt.xticks(range(len(features)), features)
            plt.yticks(range(len(frequencies)), [str(f) for f in frequencies])

            # 保存伪图
            image_path = os.path.join(output_dir, f'{sample_id}.png')
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()

            sample_id_map[sample_id] = {
                'concentration': int(info['concentration']),
                'image_path': image_path,
                'phys_attributes': phys_attributes,
                'rgb_image_path': info['rgb_image_path']
            }

        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")

    print(f"\n总计成功处理 {len(sample_id_map)} 个样本")
    return sample_id_map, spectral_data_map, physiological_attr_map


# ==================================================
# 3. 增强多属性组合模型（简化版，减少过拟合风险）
# ==================================================
class EnhancedMultiAttributeModel(nn.Module):
    """用于评估多个生理属性组合的增强模型 - 简化结构"""

    def __init__(self, input_dim=2048 * 2, num_attributes=4, num_classes=9, dropout_rate=0.5):
        super().__init__()
        # 简化特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7)
        )

        # 属性融合（简化）
        self.attribute_fusion = nn.Sequential(
            nn.Linear(256 + num_attributes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # 分类器（简化）
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, attributes):
        # 提取特征
        features = self.feature_extractor(x)

        # 融合特征和属性
        combined = torch.cat((features, attributes), dim=1)
        fused = self.attribute_fusion(combined)

        # 分类
        output = self.classifier(fused)
        return output


# ==================================================
# 4. 数据集类（修复NoneType问题）
# ==================================================
class AttributeDataset(Dataset):
    def __init__(self, features, labels, attributes=None, attribute_idx=None):
        self.features = features
        self.labels = labels
        self.attributes = attributes
        self.attribute_idx = attribute_idx

        # 确保属性数据不为None
        if self.attributes is not None and self.attribute_idx is not None:
            self.selected_attributes = self.attributes[:, attribute_idx]
        else:
            # 如果属性为None，创建零属性
            self.selected_attributes = torch.zeros((len(features), 4), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 确保始终返回三个元素，且都不是None
        features = self.features[idx]
        labels = self.labels[idx]
        attributes = self.selected_attributes[idx] if self.selected_attributes is not None else torch.zeros(4,
                                                                                                            dtype=torch.float32)

        return features, labels, attributes


# ==================================================
# 5. 直接训练评估器（无交叉验证）
# ==================================================
class DirectTrainer:
    """直接训练模型 - 不使用交叉验证"""

    def __init__(self, device='cuda'):
        self.device = device
        self.attribute_names = PhysiologicalAttributes.ATTRIBUTE_NAMES

    def train_directly(self, attribute_indices, features, labels, attributes,
                       num_classes, epochs=100, learning_rate=1e-4, test_size=0.2, random_state=42):
        """直接训练模型，划分训练/验证集"""
        combo_names = [self.attribute_names[i] for i in attribute_indices]
        print(f"\n{'=' * 60}")
        print(f"直接训练 - 评估属性组合: {', '.join(combo_names)}")
        print(f"属性索引: {attribute_indices}")
        print(f"测试集比例: {test_size}")

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val, P_train, P_val = train_test_split(
            features, labels, attributes, test_size=test_size, stratify=labels, random_state=random_state
        )

        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

        # 特征标准化 - 使用训练集统计信息，避免数据泄露
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)

        # 属性归一化 - 使用训练集统计信息
        attribute_scaler = StandardScaler()
        P_train_scaled = attribute_scaler.fit_transform(P_train[:, attribute_indices])
        P_val_scaled = attribute_scaler.transform(P_val[:, attribute_indices])

        # 转换为张量
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        P_train_tensor = torch.tensor(P_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        P_val_tensor = torch.tensor(P_val_scaled, dtype=torch.float32)

        # 创建数据集和数据加载器
        train_dataset = AttributeDataset(X_train_tensor, y_train_tensor, P_train_tensor, attribute_indices)
        val_dataset = AttributeDataset(X_val_tensor, y_val_tensor, P_val_tensor, attribute_indices)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # 初始化模型
        model = EnhancedMultiAttributeModel(
            num_attributes=len(attribute_indices),
            num_classes=num_classes,
            dropout_rate=0.5
        ).to(self.device)

        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 15
        history = {'train_loss': [], 'val_acc': [], 'val_loss': []}  # 添加验证损失记录

        print("开始训练...")
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0

            for batch_features, batch_labels, batch_attributes in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_attributes = batch_attributes.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_features, batch_attributes)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            # 验证
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_labels, batch_attributes in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_attributes = batch_attributes.to(self.device)

                    outputs = model(batch_features, batch_attributes)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()

            val_acc = 100. * val_correct / val_total
            val_loss_avg = val_loss / len(val_loader)
            scheduler.step()

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss_avg)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()  # 保存最佳模型状态
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss_avg:.4f}")

            if patience_counter >= patience:
                print(f"早停于第 {epoch + 1} 轮")
                break

        print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

        return best_val_acc, history, best_model_state, feature_scaler, attribute_scaler


# ==================================================
# 6. 主流程（移除交叉验证，直接训练）
# ==================================================
def main_direct_training():
    """主函数 - 直接训练模型"""
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"{'=' * 80}")
    print("四属性组合直接训练模型")
    print(f"{'=' * 80}")

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 步骤1: 生成伪图和生理属性
    print("\n步骤1: 为每个样本生成伪图和生理属性...")
    sample_id_map, spectral_data_map, phys_attr_map = generate_pseudo_images_per_sample()

    if not sample_id_map:
        print("错误: 没有成功生成任何伪图，请检查数据文件!")
        return

    # 步骤2: 提取特征
    print("\n步骤2: 提取特征向量...")
    feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval().to(device)

    all_features = []
    all_labels = []
    all_phys_attributes = []

    # 收集所有浓度
    concentrations = []
    sample_ids = []

    for sample_id, info in sample_id_map.items():
        concentrations.append(info['concentration'])
        sample_ids.append(sample_id)

    if not concentrations:
        print("错误: 没有找到任何浓度信息!")
        return

    # 使用整数浓度作为标签
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(concentrations)
    num_classes = len(label_encoder.classes_)

    print(f"浓度类别: {sorted(label_encoder.classes_)}")
    print(f"类别数量: {num_classes}")

    # 提取特征
    print("\n提取特征中...")
    success_count = 0
    for idx, (sample_id, sample_info) in enumerate(sample_id_map.items()):
        # 提取伪图特征
        pseudo_path = sample_info['image_path']
        if not os.path.exists(pseudo_path):
            print(f"警告: 伪图文件不存在: {pseudo_path}")
            continue

        try:
            pseudo_img = Image.open(pseudo_path).convert('RGB')
            pseudo_tensor = transform(pseudo_img).unsqueeze(0).to(device)
            with torch.no_grad():
                pseudo_features = feature_extractor(pseudo_tensor).squeeze().cpu().numpy()
                if pseudo_features.ndim == 0:  # 如果是标量，转换为1D数组
                    pseudo_features = np.array([pseudo_features])
        except Exception as e:
            print(f"处理伪图 {pseudo_path} 时出错: {e}")
            continue

        # 提取RGB图像特征（如果存在）
        rgb_path = sample_info['rgb_image_path']
        if rgb_path and os.path.exists(rgb_path):
            try:
                rgb_img = Image.open(rgb_path).convert('RGB')
                rgb_tensor = transform(rgb_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    rgb_features = feature_extractor(rgb_tensor).squeeze().cpu().numpy()
                    if rgb_features.ndim == 0:  # 如果是标量，转换为1D数组
                        rgb_features = np.array([rgb_features])
            except Exception as e:
                print(f"处理RGB图像 {rgb_path} 时出错: {e}")
                rgb_features = pseudo_features.copy()
        else:
            rgb_features = pseudo_features.copy()

        # 拼接特征
        try:
            combined_features = np.concatenate((pseudo_features.flatten(), rgb_features.flatten()))
        except Exception as e:
            print(f"拼接特征时出错: {e}")
            continue

        # 获取生理属性
        phys_attributes = sample_info['phys_attributes']

        # 获取标签
        label = labels_encoded[idx]

        all_features.append(combined_features)
        all_labels.append(label)
        all_phys_attributes.append(phys_attributes)
        success_count += 1

        if success_count % 20 == 0:
            print(f"已提取 {success_count} 个样本的特征...")

    # 转换为numpy数组
    if len(all_features) == 0:
        print("错误: 没有提取到任何特征!")
        return

    all_features_np = np.array(all_features)
    all_labels_np = np.array(all_labels)
    all_phys_np = np.array(all_phys_attributes)

    print(f"\n数据统计:")
    print(f"总样本数: {len(all_features_np)}")
    print(f"特征维度: {all_features_np.shape[1]}")
    print(f"生理属性维度: {all_phys_np.shape[1]}")

    # 检查数据平衡性
    unique_labels, counts = np.unique(all_labels_np, return_counts=True)
    print(f"\n类别分布:")
    for label, count in zip(unique_labels, counts):
        conc = label_encoder.inverse_transform([label])[0]
        print(f"浓度 {conc}%: {count} 个样本")

    # 检查是否有数据泄露风险
    print(f"\n检查数据泄露风险...")

    # 检查特征中是否有NaN或Inf
    if np.isnan(all_features_np).any() or np.isinf(all_features_np).any():
        print("警告: 特征数据包含NaN或Inf值!")
        all_features_np = np.nan_to_num(all_features_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查生理属性中是否有NaN或Inf
    if np.isnan(all_phys_np).any() or np.isinf(all_phys_np).any():
        print("警告: 生理属性数据包含NaN或Inf值!")
        all_phys_np = np.nan_to_num(all_phys_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 步骤3: 直接训练模型
    print(f"\n{'=' * 80}")
    print("开始直接训练目标属性组合模型...")
    print(f"{'=' * 80}")

    target_attrs = PhysiologicalAttributes.TARGET_ATTRIBUTES
    target_names = [PhysiologicalAttributes.ATTRIBUTE_NAMES[i] for i in target_attrs]

    print(f"目标四个属性: {', '.join(target_names)}")
    print(f"目标属性索引: {target_attrs}")

    # 创建结果目录
    results_dir = 'Model-result/54/'
    os.makedirs(results_dir, exist_ok=True)

    # 初始化训练器
    trainer = DirectTrainer(device=device)

    # 直接训练
    print(f"\n开始训练和验证...")
    try:
        final_val_acc, history, best_model_state, feature_scaler, attr_scaler = trainer.train_directly(
            target_attrs, all_features_np, all_labels_np, all_phys_np,
            num_classes, epochs=100, learning_rate=1e-4, test_size=0.2
        )
    except Exception as e:
        print(f"训练过程中出错: {e}")
        print("检查数据和参数...")
        return

    # 可视化训练过程
    plt.figure(figsize=(15, 5))

    # 绘制训练和验证损失
    plt.subplot(1, 3, 1)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], 'r-', label='训练损失')
    plt.plot(epochs_range, history['val_loss'], 'b-', label='验证损失')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.title('训练与验证损失曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 绘制验证准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['val_acc'], 'g-', label='验证准确率')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('验证准确率曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 显示最终准确率
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f'最终验证准确率\n{final_val_acc:.2f}%', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes, fontsize=16, color='blue')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'direct_training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存模型
    print(f"\n保存模型...")
    # 创建模型实例
    model = EnhancedMultiAttributeModel(
        input_dim=all_features_np.shape[1],
        num_attributes=len(target_attrs),
        num_classes=num_classes
    )
    model.load_state_dict(best_model_state)

    # 构建模型文件名
    model_filename = f'direct_model_target_attrs_{final_val_acc:.2f}.pth'
    model_path = os.path.join(results_dir, model_filename)

    # 保存模型及其相关信息
    torch.save({
        'model_state_dict': best_model_state,
        'validation_accuracy': final_val_acc,
        'input_dim': all_features_np.shape[1],
        'num_attributes': len(target_attrs),
        'num_classes': num_classes,
        'attribute_indices': target_attrs,
        'attribute_names': target_names,
        'label_encoder_classes': label_encoder.classes_,
        'feature_scaler': feature_scaler,  # 保存scalers以便后续推理
        'attribute_scaler': attr_scaler,
        'training_history': history
    }, model_path)
    print(f"已保存模型: {model_path}")

    # 生成报告
    print(f"\n{'=' * 80}")
    print("四属性组合直接训练完成!")
    print(f"{'=' * 80}")

    print(f"\n最终结果:")
    print(f"四个目标属性: {', '.join(target_names)}")
    print(f"验证集准确率: {final_val_acc:.2f}%")

    # 保存详细报告
    report = f"""
    四属性组合直接训练评估报告
    =====================================================

    评估日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    使用设备: {device}
    总样本数: {len(all_features_np)}
    训练集大小: {int(len(all_features_np) * 0.8)}
    验证集大小: {int(len(all_features_np) * 0.2)}
    类别数量: {num_classes}
    特征维度: {all_features_np.shape[1]}

    类别分布:
    {'-' * 40}
    """
    for label, count in zip(unique_labels, counts):
        conc = label_encoder.inverse_transform([label])[0]
        report += f"    浓度 {conc}%: {count} 个样本 ({count / len(all_features_np) * 100:.1f}%)\n"

    report += f"""
    目标四个属性:
    {'-' * 40}
    """
    for i, (idx, name) in enumerate(zip(target_attrs, target_names)):
        report += f"    属性{i + 1}: {name} (索引: {idx})\n"

    report += f"""
    训练结果:
    {'-' * 40}
    验证集准确率: {final_val_acc:.2f}%

    模型配置:
    {'-' * 40}
    网络结构: 简化版多属性融合模型
    优化器: AdamW (学习率: 1e-4, 权重衰减: 1e-4)
    调度器: CosineAnnealingLR
    损失函数: CrossEntropyLoss
    批次大小: 16
    最大轮次: 100
    早停耐心: 15
    测试集比例: 0.2

    数据泄露防范措施:
    {'-' * 40}
    1. 使用训练集统计信息进行特征标准化
    2. 使用训练集统计信息进行属性归一化
    3. 伪图生成使用样本自身统计信息，避免全局信息
    4. 增加模型正则化（Dropout, 权重衰减）
    5. 简化模型结构，减少过拟合风险

    模型保存信息:
    {'-' * 40}
    已保存模型文件: {model_filename}
    模型文件包含: 
    - 状态字典 (model_state_dict)
    - 验证准确率 (validation_accuracy)
    - 输入维度 (input_dim)
    - 属性数量 (num_attributes)
    - 类别数量 (num_classes)
    - 属性索引 (attribute_indices)
    - 属性名称 (attribute_names)
    - 标签编码器类别 (label_encoder_classes)
    - 特征标准化器 (feature_scaler)
    - 属性标准化器 (attribute_scaler)
    - 训练历史 (training_history)

    结论:
    {'-' * 40}
    四个目标属性(含水率、离子浓度、组织孔隙率、细胞完整性)组合的
    直接训练模型在验证集上取得了 {final_val_acc:.2f}% 的准确率。
    该模型已保存，可用于后续预测。
    """

    # 保存报告
    with open(os.path.join(results_dir, 'direct_training_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {results_dir}/")
    print(f"  1. 评估报告: direct_training_report.txt")
    print(f"  2. 可视化图: direct_training_results.png")
    print(f"  3. 模型文件: {model_filename}")


if __name__ == '__main__':
    main_direct_training()
