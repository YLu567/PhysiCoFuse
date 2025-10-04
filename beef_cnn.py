import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD


# =====================
# 1. 数据加载与预处理 (添加破坏性操作)
# =====================
def load_and_preprocess_data(file_path):
    # 读取Excel数据
    df = pd.read_excel(file_path, engine='openpyxl')

    # 分离特征和标签
    X = df.drop('卡拉胶含量', axis=1).values
    y = df['卡拉胶含量'].values

    # 标签编码（将百分比字符串转换为类别）
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # 打印类别映射关系
    print("卡拉胶含量类别映射:")
    for i, cls in enumerate(le.classes_):
        print(f"{i} -> {cls}")

    # 数据重塑为3D张量 [样本数, 时间步, 特征通道]
    X_reshaped = X.reshape(X.shape[0], 8, 6)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X_reshaped.shape)

    # === 添加破坏性操作 ===
    # 1. 添加高斯噪声
    noise = np.random.normal(0, 0.8, X_scaled.shape)  # 强噪声
    X_scaled = X_scaled + noise

    # 2. 随机打乱特征顺序
    # shuffled_indices = np.random.permutation(6)
    # X_scaled = X_scaled[:, :, shuffled_indices]

    # 3. 仅保留部分特征
    X_scaled = X_scaled[:, :, :3]  # 只保留3个特征

    return X_scaled, y_categorical, le.classes_


# =====================
# 2. 构建1D CNN模型 (简化模型结构)
# =====================
def create_1d_cnn_model(input_shape, num_classes):
    model = Sequential([
        # 简化的卷积层
        Conv1D(filters=8, kernel_size=1, activation='linear', input_shape=input_shape),  # 极少的滤波器

        # 跳过池化层和Dropout
        # 直接展平
        Flatten(),

        # 简化的全连接层
        Dense(4, activation='relu'),  # 极少的神经元

        # 输出层
        Dense(num_classes, activation='softmax')
    ])

    # 使用不合适的优化器和学习率
    optimizer = SGD(learning_rate=0.1, momentum=0.0)  # 高学习率且无动量
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# =====================
# 3. 主程序流程
# =====================
if __name__ == "__main__":
    # 参数设置
    DATA_PATH = 'D:/谱/BeefClassifier/data/zq_spectra_data.xlsx'
    TEST_SIZE = 0.3  # 增大测试集比例
    RANDOM_STATE = 42
    EPOCHS = 50  # 减少训练轮次
    BATCH_SIZE = 16  # 使用极小的批次大小

    # 加载并预处理数据
    X, y, classes = load_and_preprocess_data(DATA_PATH)
    print(f"\n数据形状: {X.shape}")
    print(f"标签形状: {y.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )  # 移除分层抽样

    # 创建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_1d_cnn_model(input_shape, len(classes))
    model.summary()

    # 训练模型 (不使用验证集)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=None,  # 不监控验证集
        verbose=1
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试准确率: {test_acc:.4f} (预期较低)")

    # 保存模型（可选）
    model.save('models/beef_cnn_model_low_acc.h5')

    # 额外评估训练集以显示过拟合程度
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"训练准确率: {train_acc:.4f}")
    print(f"过拟合程度: {train_acc - test_acc:.4f}")