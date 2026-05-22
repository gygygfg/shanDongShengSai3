# 山东大学生竞赛 —— 图像处理三赛道

> 🏆 **山东省大学生竞赛项目** · 涵盖图像去噪、指纹修复、纤维识别三大任务

---

## 📂 项目总览

本仓库包含三个独立的图像处理算法项目，分别对应竞赛中的三个任务赛道。每个项目都提供了完整的数学模型推导、Python 实现代码和可复现的实验结果。

```
dasai/
├── 1Den/                          ← 输入：10张含噪干涉条纹图（01~10.png）
├── 2Rec/                          ← 输入：10张模糊指纹图（11~20.png）
├── 3Seg/                          ← 输入：10张显微镜灰度图（21~30.png）
├── results/                       ← 输出：处理结果（已去噪的条纹图）
├── generated/                     ← 生成：纤维检测的合成数据
│
├── interferenceFringeDenoising/   ─ 赛道一：干涉条纹去噪
├── fingerprintRestoration/        ─ 赛道二：指纹图像增强与修复
├── ossifyingFibroma/              ─ 赛道三：显微镜纤维识别
│
├── .gitignore                     ─ Git 忽略规则（排除大文件/缓存）
└── README.md                      ─ 本文件
```

---

## 赛道一 🧪 干涉条纹去噪

**目录：** `interferenceFringeDenoising/`

### 任务目标

对含噪声的干涉条纹图像（1Den/01~10.png）进行去噪处理，恢复清晰的条纹结构，输出二值化结果到 `results/` 目录。

### 技术方案

| 方法 | 说明 |
|------|------|
| **U-Net 深度学习去噪** | 使用 PyTorch 实现经典 U-Net 架构，在合成条纹数据上训练去噪模型 |
| **CLAHE 自适应对比度增强** | 对偏暗图像进行对比度拉伸，提升模型去噪效果 |
| **Otsu 自动二值化** | 模型输出概率图后，使用 Otsu 算法自适应确定二值化阈值 |

### 文件说明

| 文件 | 用途 |
|------|------|
| `main.py` | 主推理脚本 — 加载 U-Net 模型，对 1Den/ 下的图片逐张去噪，输出到 results/ |
| `train_unet.py` | U-Net 模型定义 + 训练脚本（DoubleConv → Down → Up → Out 架构） |
| `generate_fringes.py` | 合成条纹数据生成器 — 生成直线形、圆形、双核型、波纹型干涉条纹，添加高斯+散斑噪声 |
| `pyproject.toml` | 项目配置与依赖声明 |
| `checkpoints/` | 训练好的模型权重（`.pth` 文件，被 .gitignore 排除） |
| `dataset/` | 测试样本数据 |

### 运行方式

```bash
cd interferenceFringeDenoising

# 生成合成训练数据（10000张256x256条纹图）
python generate_fringes.py

# 训练 U-Net 模型
python train_unet.py

# 对 1Den/ 图片去噪并输出到 ../results/
python main.py
```

---

## 赛道二 🖐️ 指纹图像增强与修复

**目录：** `fingerprintRestoration/`

### 任务目标

对低质量、模糊不清的指纹图像（2Rec/11~20.png）进行增强处理，恢复清晰的脊线结构，提升指纹的可识别性。

### 技术方案

采用**四步流水线**处理：

```
输入图像 → 指纹分割 → 方向场估计 → 脊线频率估计 → Gabor滤波增强 → 二值化 → 输出图像
```

| 模块 | 方法 | 说明 |
|------|------|------|
| **GMFS** | 梯度幅值指纹分割 | 利用 Sobel 梯度幅值 + 自适应阈值分离指纹与背景 |
| **GBFOE** | 梯度方向场估计 | 基于梯度协方差矩阵估算脊线局部方向 |
| **XSFFE** | X 签名频率估计 | 沿脊线垂直方向投影，检测周期性的波峰/波谷 |
| **GBFEN** | Gabor 滤波增强 | 构建 16 方向 × 9 周期 = 144 个 Gabor 滤波器核，上下文相关滤波 |
| **Otsu** | 大津二值化 | 最大化类间方差，自适应确定二值化阈值 |

### 文件说明

| 文件 | 用途 |
|------|------|
| `main.py` | 主处理脚本 — 调用 pyfing 和 fingerprint_enhancer 库完成增强 |
| `README.md` | 详细算法文档（含完整数学公式推导） |
| `pyproject.toml` | 项目配置 |

### 依赖库

- `pyfing` — 提供 GMFS、GBFOE、XSFFE、GBFEN 方法
- `fingerprint-enhancer` — 方向 Gabor 滤波器组增强
- `OpenCV`、`NumPy`、`TensorFlow/Keras`

### 运行方式

```bash
cd fingerprintRestoration
python main.py
```

---

## 赛道三 🔬 显微镜高噪声纤维识别

**目录：** `ossifyingFibroma/`

### 任务目标

在**高噪声背景**的显微灰度图像（3Seg/21~30.png）中识别极弱神经元纤维（灰度仅 20-40，淹没在噪声中）。传统全局二值化、高斯差分等方法完全失效。

### 技术方案

采用**四种互补的数学模型 + 加权融合 + 后处理管线**：

```
输入图像 → [频域带通 + Gabor + Top-Hat + Hessian] → 加权融合 → 二值化 → 形态学清理 → 伸长率过滤 → PCA趋势降噪 → 骨架化 → 输出
```

| 检测方法 | 原理 | 权重 |
|----------|------|:----:|
| **① 频域带通滤波** | Butterworth 带通滤波器（FFT），保留纤维中频方向信息 | 0.25 |
| **② Gabor 方向滤波器组** | 6个方向 Gabor 核，逐像素取最大响应 | 0.35 |
| **③ Top-Hat 形态学变换** | 多角度椭圆结构元素，提取细长亮/暗线 | 0.20 |
| **④ 多尺度 Hessian 脊线检测** | Hessian 矩阵特征值分析 + Frangi 线状因子，5个尺度融合 | 0.20 |

**后处理管线：** Otsu/分位数二值化 → 形态学开闭运算 → 伸长率过滤（保留细长结构）→ PCA 主成分趋势降噪 → 距离变换骨架化

### 文件说明

| 文件/目录 | 用途 |
|-----------|------|
| `main.py` | 主程序 — 数据生成 + 四大方法检测 + 融合 + 后处理完整管线 |
| `modules/__init__.py` | 模块包声明 |
| `modules/processing.py` | 图像处理管线：二值化、形态学、骨架化、端点桥接等 |
| `README.md` | 详细算法文档（含完整数学公式推导） |
| `pyproject.toml` | 项目配置 |

### 运行方式

```bash
cd ossifyingFibroma

# 运行完整纤维识别管线
python main.py

# 生成高噪声弱纤维训练数据
python main.py --generate
python main.py --generate --num 500

# 指定检测方法
python main.py --method bandpass   # 仅频域带通
python main.py --method gabor      # 仅 Gabor 方向滤波
python main.py --method tophat     # 仅 Top-Hat 形态学
python main.py --method hessian    # 仅 Hessian 脊线检测
python main.py --method all        # 融合所有方法（默认）
```

---

## 📊 数据集说明

| 目录 | 数量 | 内容 | 用途 |
|------|:----:|------|------|
| `1Den/` | 10张 | 含噪干涉条纹（01.png ~ 10.png） | 去噪任务输入 |
| `2Rec/` | 10张 | 低质量指纹图像（11.png ~ 20.png） | 指纹修复任务输入 |
| `3Seg/` | 10张 | 高噪声显微镜灰度图（21.png ~ 30.png） | 纤维识别任务输入 |
| `results/` | 10张 | 去噪后二值条纹图（1Den_01.png ~ 1Den_10.png） | 去噪任务输出 |
| `generated/` | - | 合成纤维数据（images + masks） | 纤维检测辅助数据 |

---

## 🛠 技术栈

| 工具/库 | 用途 |
|---------|------|
| **Python 3.12+** | 开发语言 |
| **PyTorch** | 深度学习框架（U-Net） |
| **OpenCV** | 图像处理：滤波、形态学、Gabor、阈值等 |
| **NumPy / SciPy** | 数值计算、FFT、统计、特征分解 |
| **scikit-image** | 骨架化等高级形态学操作 |
| **pyfing** | 指纹专用处理库 |
| **fingerprint-enhancer** | 指纹 Gabor 增强库 |
| **uv** | 项目依赖管理 |

---

## 📝 License

MIT License
