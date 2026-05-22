# CNN + GAN 混合模型：骨化性纤维瘤图像语义分割

## 项目概述

本项目使用 **CNN + GAN 混合框架** 对医学图像中的骨化性纤维瘤区域进行语义分割。从 `../3Seg/` 原图中识别 `./3Seg_Labeled/` 中红色标注的区域，输出二值分割 mask。

## 数学模型与工具

### 1. 网络架构

#### (1) Generator — U-Net 风格分割网络 (`TimmUNetGenerator`)
- **编码器（Encoder）**：使用 `timm` (PyTorch Image Models) 库加载 **EfficientNet-B0** 预训练 CNN 作为骨干网络，提取多尺度特征图（`features_only=True`）。
- **解码器（Decoder）**：逐层上采样（`F.interpolate`，双线性插值 `bilinear`）+ **跳跃连接（Skip Connection）**，将深层语义信息与浅层细节信息融合。
- **输出头**：`Conv2d(32→1)` 输出单通道 logits，由损失函数内部的 sigmoid 处理为概率。

#### (2) Discriminator — PatchGAN 判别器 (`PatchGANDiscriminator`)
- 输入：原图（3 通道）与 mask（1 通道）拼接 → **4 通道输入**
- 架构：4 层卷积（`kernel_size=4, stride=2`）+ `LeakyReLU(0.2)` 下采样，输出单通道 `[B, 1, H', W']` 的 **PatchGAN 判别图**，对每个图像块（patch）独立判断真伪。
- 逐 patch 判别机制能更精细地约束分割边缘，尤其适合枝状细结构的分割任务。

### 2. 损失函数（Loss Functions）

| 损失函数 | 数学形式 | 作用 |
|---------|---------|------|
| **Focal Loss** | `FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)` <br> α=0.75, γ=2.0 | 自动聚焦难分样本，缓解正负样本不均衡，对细/枝状结构更友好 |
| **Dice Loss** | `DL = 1 - (2·\|P∩T\| + ε) / (\|P\| + \|T\| + ε)` | 度量预测 mask 与 GT mask 的重叠度，直接优化分割交并比 |
| **面积正则化（Area Regularization）** | `λ · mean(pred_prob)` , λ=0.1 | L1 惩罚预测 mask 的总体积，防止模型过度膨胀"一片" |
| **对抗损失（GAN Loss）** | `L_adv = MSE(D(x, y_real), 1) + MSE(D(x, y_fake), 0)` | **LSGAN（Least Squares GAN）**，使用 MSE 替代传统 BCE，训练更稳定 |

**组合分割损失**：
```
L_seg = FocalLoss(pred, target) + DiceLoss(sigmoid(pred), target) + 0.1 · mean(sigmoid(pred))
```

**Generator 总损失**：
```
L_G = L_seg + 0.05 · L_adv
```

### 3. 优化策略

- **优化器（Optimizer）**：`Adam`，`betas=(0.5, 0.999)`
  - Generator 学习率：`1e-4`
  - Discriminator 学习率：`4e-4`（比 Generator 高，保持判别器竞争力）
- **学习率调度（Scheduler）**：`CosineAnnealingLR(T_max=NUM_EPOCHS)` — 余弦退火调整学习率

### 4. 评估指标

- **IoU（Intersection over Union）**：`IoU = |P ∩ T| / |P ∪ T|`，衡量分割区域与真实区域的重叠程度
- **Dice Coefficient**：`Dice = 2|P ∩ T| / (|P| + |T|)`，与 IoU 正相关但更平滑

推理阈值设为 `0.75`（高于常规的 0.5），以抑制过度分割。

### 5. 数据预处理

- **红色区域提取**：通过颜色阈值 `(R > 150) & (G < 80) & (B < 80)` 从 RGB 标注图中提取二值 mask
- **数据增强**：
  - 随机水平翻转（概率 0.5）
  - 随机旋转（-15° ~ +15°，双线性插值）

## 环境依赖

| 工具/库 | 用途 |
|---------|------|
| **PyTorch** | 深度学习框架，构建 CNN + GAN 模型 |
| **timm** | 提供 EfficientNet-B0 等预训练 CNN backbone |
| **torchvision** | 图像变换（Resize、ToTensor） |
| **Pillow (PIL)** | 图像读取与处理 |
| **NumPy** | 数组运算与 mask 处理 |
| **tqdm** | 训练进度条可视化 |
| **CUDA** | GPU 加速（自动检测，fallback 到 CPU） |

## 训练流程

```
1. 数据加载 → SegmentationDataset（读取 ../3Seg/ 和 ./3Seg_Labeled/）
2. 数据增强（翻转 + 旋转）
3. 红色区域提取 → 二值 mask
4. 训练 Discriminator（最小化 GAN Loss）
5. 训练 Generator（最小化 分割损失 + 对抗损失）
6. CosineAnnealingLR 更新学习率
7. 验证评估（IoU / Dice）
8. 保存最佳模型（基于 IoU）
9. 推理：对所有测试图预测，保存二值 mask
```

## 项目结构

```
ossifyingFibroma/
├── main.py               # 主程序（训练 + 推理）
├── 3Seg_Labeled/         # 红色标注图（GT）
├── outputs/              # 模型保存（best_model.pth）
└── README.md             # 本文档

../3Seg/                  # 原图
../results/               # 推理结果
```

## 运行方式

```bash
cd ossifyingFibroma
python main.py
```
