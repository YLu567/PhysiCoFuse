# 🥩 PhysiCoFuse: 物理引导的阻抗与视觉融合框架
**Biophysically-Guided Spatially Consistent Alignment of Electrical Impedance Spectroscopy and RGB Imagery**

本仓库包含 PhysiCoFuse 模型的完整训练代码，用于基于 EIS（电阻抗谱）和 RGB 图像的多模态融合检测肉类中亲水胶体掺假。
---

 **PhysiCoFuse** 框架包含三个核心模块：
1.  **多频率阻抗编码器 (MIE)**：从原始EIS数据中提取判别性特征，并自适应地强调关键频率信息。。
2.  **伪生理图生成器 (PPMG)**：将抽象的一维EIS信号转化为空间结构化的“伪图像”，作为连接电学信号与视觉特征的桥梁。
3.  **跨模态物理一致性模块 (CMPC)**：在共享物理空间中强制对齐视觉特征与电学特征，确保多模态融合基于共同的物理本质，而非单纯的统计相关性。

---


## 🏗️ 框架架构 (Framework Architecture)

PhysiCoFuse 的核心逻辑包含以下步骤：
使用 EIS (8×6) 和 RGB 图像 作为双模态输入
包含 PPMG（伪生理图生成器），生成 4 通道伪生理属性图
7 通道 ResNet50（RGB + 4 伪图）提取图像特征
MIE（多频阻抗编码器） + CMPC（跨模态物理一致性模块）
5 折 Stratified 交叉验证
数据增强（仅训练集）
混合精度训练（自动混合精度，仅在 GPU 可用时启用）
半精度模型保存（float16），显著减小模型文件大小
自动保存原始 EIS 伪图（每折保存 4 张）

---

## 🛠️ 环境配置 (Environment Setup)

本项目基于 Python 开发，建议使用 Conda 或 Virtualenv 管理环境。

### 1. 依赖库 (Dependencies)
```bash
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
Pillow>=9.0.0
openpyxl>=3.0.0   # 读取 .xlsx
matplotlib>=3.5.0
tqdm>=4.64.0
