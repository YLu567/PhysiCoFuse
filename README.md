# 🥩 PhysiCoFuse: 物理引导的阻抗与视觉融合框架
**Biophysically-Guided Spatially Consistent Alignment of Electrical Impedance Spectroscopy and RGB Imagery**

本仓库提供了 PhysiCoFuse 框架的官方实现。这是一个基于深度学习的多模态融合框架，旨在通过结合**电阻抗谱 (EIS)** 和 **RGB 图像**，实现对肉制品中亲水胶体（如卡拉胶）掺假的无损、高精度定量检测。

该框架通过引入生物物理约束，解决了传统多模态融合中缺乏物理一致性和空间对齐的痛点，实现了 SOTA (State-of-the-Art) 的检测精度。

---

 **PhysiCoFuse** 框架包含三个核心模块：
1.  **多频率阻抗编码器 (MIE)**：提取关键频段特征。
2.  **伪生理图生成器 (PPMG)**：将 EIS 光谱编码为具有物理语义（如含水量、离子浓度）的空间图。
3.  **跨模态物理一致性模块 (CMPC)**：强制约束介电特征与光学特征之间的物理一致性。

---


## 🏗️ 框架架构 (Framework Architecture)

PhysiCoFuse 的核心逻辑包含以下步骤：

1.  **数据获取**：同步采集牛肉样本的 RGB 图像和 EIS 数据（8个频率点 × 6个电气参数）。
2.  **多频率阻抗编码 (MIE)**：利用频率注意力机制，自适应加权对掺假检测最有用的频率信息。
3.  **伪生理图生成 (PPMG)**：
    *   将一维阻抗特征解码为水含量、离子浓度等生理属性。
    *   将这些属性扩展为空间热力图（Pseudo-Physiological Map），与 RGB 特征图对齐。
4.  **跨模态物理一致性 (CMPC)**：
    *   强制 RGB 分支预测的物理属性与 EIS 分支预测的属性保持一致（通过 MSE Loss 约束）。
5.  **特征融合与分类**：融合后的特征输入 MLP 进行最终分类。

---

## 🛠️ 环境配置 (Environment Setup)

本项目基于 Python 开发，建议使用 Conda 或 Virtualenv 管理环境。

### 1. 依赖库 (Dependencies)
```bash
python==3.11
torch==2.0.1
torchvision==0.15.2
numpy
pandas
scikit-learn
opencv-python
matplotlib
