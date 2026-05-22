# 显微镜灰度图 — 高噪声背景下神经元纤维识别系统

**Neuron Fiber Detection under Heavy Noise in Microscopy**

---

## 问题概述

| 痛点 | 说明 |
|------|------|
| 背景噪声大 | 显微成像噪声 + 光照不均 + 组织纹理 |
| 纤维极弱 | 神经元纤维灰度仅 20-40，淹没在噪声中 |
| 传统方法失效 | 全局二值化、高斯模糊+差分均无法检测 |

---

## 数学模型与技术架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    神经元纤维识别管线                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─ 数据生成 ──────────────────────────────────────────────────┐    │
│  │  Butterworth 带通纹理 (频域) + 光照不均 + 泊松-高斯噪声     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│  ┌─ 四大纤维检测模型 ─────────────────────────────────────────┐    │
│  │  ① 频域带通滤波 (Butterworth)     ② Gabor 方向滤波器组     │    │
│  │  ③ Top-Hat 形态学变换             ④ Hessian 脊线检测       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│  ┌─ 加权融合 ──────────────────────────────────────────────────┐    │
│  │  fused = Σ w_i · R_i, 归一化到 [0,1]                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│  ┌─ 后处理管线 ────────────────────────────────────────────────┐    │
│  │  二值化 → 形态学清理 → 伸长率过滤 → PCA趋势降噪 → 骨架化    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│                         二值 Mask                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 一、数据生成模块 — 模拟真实显微组织纹理

### 1.1 Butterworth 带通纹理

**数学模型：**

对高斯白噪声进行频域滤波，构造各向同性纹理：

```
噪声:        N(x,y) ~ N(0, 1)
FFT:         F(ω) = ℱ{N(x,y)}
Butterworth: H(ω) = LP(ω) · HP(ω)

低通:  LP(ω) = 1 / (1 + (||ω|| / ω_high)^(2n))
高通:  HP(ω) = 1 - 1 / (1 + (||ω|| / ω_low)^(2n))

纹理:  T(x,y) = ℱ⁻¹{F(ω) · H(ω)} × (σ_target / σ_current)
```

其中 `ω_low = 0.015`（滤除大面积不均匀），`ω_high = 0.22`（滤除像素噪声），`n = 1.8`（过渡带陡峭度）。

**工具：** `numpy.fft.fft2`, `numpy.fft.ifft2`, `numpy.fft.fftshift`, `numpy.ogrid`

### 1.2 光照不均模拟

**数学模型：** 多个随机高斯斑点叠加

```
Illumination(x,y) = Σᵢ Aᵢ · exp(-||(x,y) - μᵢ||² / 2σᵢ²)
```

高斯斑点经平滑后叠加到纹理上，模拟真实显微成像中因组织厚薄不均导致的亮度变化。

**工具：** `cv2.GaussianBlur`

### 1.3 显微传感噪声

**数学模型：** 泊松-高斯混合噪声

```
Noisy(I) = Poisson(I · λ) / λ + N(0, σ²)
         ↑ 光子计数噪声         ↑ 读出噪声
```

泊松噪声模拟光子计数统计（信号相关），高斯噪声模拟传感器读出噪声（信号独立）。

**工具：** `numpy.random.poisson`, `numpy.random.normal`

---

## 二、纤维检测数学模型

### 2.1 频域带通滤波 (Butterworth Bandpass)

**数学原理：** 纤维在图像中表现为具有一定宽度和方向的中频结构。通过频域带通滤波，抑制极低频（光照不均）和极高频（随机噪声），保留纤维的中频方向性信息。

**Butterworth 带通滤波器：**

```
HP(ω) = 1 / (1 + (ω_low / (||ω|| + ε))^(2n))     高通 (抑制低频)
LP(ω) = 1 / (1 + (||ω|| / ω_high)^(2n))           低通 (抑制高频)
H(ω)  = HP(ω) · LP(ω)                              带通

参数: ω_low = 0.02, ω_high = 0.35, n = 2
```

**流程：** 输入图像 → FFT → 频域中心化 → 乘以 Butterworth 掩模 → 逆FFT → 取模 → 归一化

**工具：** `numpy.fft.fft2`, `numpy.fft.ifft2`, `numpy.fft.fftshift`, `numpy.fft.ifftshift`

### 2.2 Gabor 方向滤波器组

**数学原理：** Gabor 滤波器是由高斯包络调制的正弦平面波，在空域和频域均具有最优的联合分辨率。对特定方向和频率的线状结构产生强响应。

**Gabor 核定义：**

```
g(x,y; λ, θ, ψ, σ, γ) = exp(-(x'² + γ²y'²) / 2σ²) · cos(2πx'/λ + ψ)

其中:
  x' = x·cosθ + y·sinθ
  y' = -x·sinθ + y·cosθ
  λ = 波长 (10.0)         γ = 空间长宽比 (0.5)
  θ = 方向角              σ = 高斯包络标准差 (4.0)
  ψ = 相位偏移 (0)
```

**方向融合：** 6 个方向 `θ ∈ {0, π/6, π/3, π/2, 2π/3, 5π/6}`，逐像素取最大值响应：

```
R(x,y) = max_{θ∈Θ} |I(x,y) ∗ g_θ(x,y)|
```

**工具：** `cv2.getGaborKernel`, `cv2.filter2D`

### 2.3 Top-Hat 形态学变换

**数学原理：** 使用数学形态学开/闭运算，从局部邻域中提取细长结构。即使纤维灰度绝对值不高，只要在局部是"最亮/最暗"的细长结构，就能被分离。

**Top-Hat (亮线)：**

```
TopHat(A) = A - (A ⊖ B) ⊕ B
          = A - Opening(A, B)
```

**Black-Hat (暗线)：**

```
BlackHat(A) = (A ⊕ B) ⊖ B - A
            = Closing(A, B) - A
```

**多角度增强：** 将细长椭圆结构元素旋转 6 个方向（0°~180°），逐像素取最大响应，解决单一方向对倾斜纤维响应弱的问题。

**结构元素：** 椭圆核 31×3（长轴沿纤维方向，短轴略宽于纤维线宽）

**工具：** `cv2.morphologyEx(MORPH_TOPHAT / MORPH_BLACKHAT)`, `cv2.getStructuringElement(MORPH_ELLIPSE)`, `cv2.warpAffine`

### 2.4 多尺度 Hessian 脊线检测

**数学原理：** 利用 Hessian 矩阵的特征值分析，区分纤维状结构（脊线/谷线）和团状结构。纤维像素的二阶偏导表现出"一个特征值大、另一个接近零"的特性。

**Hessian 矩阵：**

```
H(x,y) = [[Ixx, Ixy],
          [Ixy, Iyy]]

其中 Ixx = ∂²I/∂x², Iyy = ∂²I/∂y², Ixy = ∂²I/∂x∂y
```

**特征值计算：**

```
trace = Ixx + Iyy
det   = Ixx·Iyy - Ixy²
disc  = √(max(trace² - 4·det, 0))

λ₁ = (trace + disc) / 2    (λ₁ ≥ λ₂)
λ₂ = (trace - disc) / 2
```

**脊线强度 (亮纤维模式)：**

```
strength = max(-λ₂, 0)          ← 仅响应亮脊线
ratio    = |λ₁| / (|λ₂| + ε)    ← 线状 vs 团状
line_factor = exp(-ratio² / 2β²)  ← Frangi 线状因子
response = strength × line_factor
```

**多尺度融合：** 逐像素取各尺度 σ ∈ {1.0, 2.0, 3.0, 4.0, 5.0} 的最大响应，以匹配不同粗细的纤维。

**工具：** `scipy.ndimage.sobel`, `scipy.ndimage.gaussian_filter`, `numpy.sqrt`

---

## 三、融合与后处理

### 3.1 加权融合

**数学模型：**

```
fused(x,y) = Σᵢ wᵢ · Rᵢ(x,y)    →    归一化到 [0,1]

权重:
  w_bandpass = 0.25    (频域带通)
  w_gabor    = 0.35    (Gabor 方向滤波)
  w_tophat   = 0.20    (Top-Hat 形态学)
  w_hessian  = 0.20    (Hessian 脊线检测)
```

### 3.2 二值化

| 方法 | 公式 | 说明 |
|------|------|------|
| Otsu | `argmax_t σ²_between(t)` | 自动阈值，最大化类间方差 |
| Mean 分位数 | `t = μ + σ·Φ⁻¹(1-α)` | 基于正响应像素的均值和标准差，取上 α 分位数 |
| Fixed | `t = τ·255` | 固定阈值 |

**工具：** `cv2.threshold`, `scipy.stats.norm.ppf`

### 3.3 形态学清理

闭运算连接纤维缝隙 → 开运算去除孤立噪点：

```
Clean(A) = (A ⊕ B₁) ⊖ B₁  →  闭运算 (核 5×5, 2次迭代)
         → (A ⊖ B₂) ⊕ B₂  →  开运算 (核 3×3, 1次迭代)
```

**工具：** `cv2.morphologyEx(MORPH_CLOSE / MORPH_OPEN)`, `cv2.getStructuringElement(MORPH_ELLIPSE)`

### 3.4 伸长率过滤

**数学模型：** 对每个连通域，计算像素坐标的协方差矩阵，通过特征值比值衡量细长程度。

**协方差矩阵：**

```
Σ = [[σxx,  σxy],
     [σxy,  σyy]]

σxx = Σ(xᵢ - x̄)²/n,  σyy = Σ(yᵢ - ȳ)²/n
σxy = Σ(xᵢ - x̄)(yᵢ - ȳ)/n
```

**伸长率：**

```
λ₁₂ = (σxx + σyy ± √((σxx - σyy)² + 4σxy²)) / 2
elongation = √(λ₁ / λ₂)    (λ₁ ≥ λ₂ > 0)

圆形 ≈ 1.0,  纤维 ≫ 1.0
保留条件: area ≥ 15 且 elongation ≥ 1.5
```

**工具：** `numpy.sqrt`, `scipy.ndimage.label`

### 3.5 PCA 趋势线降噪

**数学原理：** 对主纤维组件的所有像素点进行 PCA（主成分分析），拟合趋势线方向。以趋势线为基准路径，保留路径附近的组件，消除远离路径的孤立噪声。

**PCA 拟合：**

```
均值:     μ = (x̄, ȳ)
中心化:   X_centered = {(xᵢ - x̄, yᵢ - ȳ)}
协方差:   C = (1/n) · X_centeredᵀ · X_centered
特征分解: C · v = λ · v
主方向:   v₁ = argmax(λ) 对应特征向量
趋势线:   a·x + b·y + c = 0, (a,b) ⟂ v₁
```

**自适应距离阈值：** 基于主组件像素到趋势线的 P95 残差 × 5.0（下限 10px，上限 300px），跟随组件自身宽度变化。

**工具：** `numpy.cov`, `numpy.linalg.eigh`, `numpy.percentile`

### 3.6 骨架化

**数学模型：** 将离散纤维片段聚合成连续线。

1. **自适应闭运算：** 核大小 = 0.02 × min(h, w)，连接断裂处
2. **距离变换：** `d(x,y) = min_{(x',y')∈背景} ||(x,y)-(x',y')||₂`，记录原始宽度半径
3. **形态学骨架化：** 逐层剥离边缘像素至单像素宽，保留拓扑结构
4. **修剪短分支：** 过滤少于 `MIN_SKELETON_LENGTH` 个节点的骨架
5. **恢复宽度：** 骨架每点以距离变换值为半径画圆盘

**工具：** `cv2.distanceTransform(DIST_L2)`, `skimage.morphology.skeletonize`, `cv2.circle`

---

## 技术栈

| 工具/库 | 用途 | 关键函数/类 |
|---------|------|-------------|
| **Python 3.12+** | 开发语言 | — |
| **NumPy** | 矩阵运算、FFT、特征分解、统计 | `numpy.fft`, `numpy.linalg.eigh`, `numpy.cov`, `numpy.percentile`, `numpy.ogrid`, `numpy.random` |
| **OpenCV (cv2)** | Gabor 滤波、形态学、距离变换、I/O | `cv2.getGaborKernel`, `cv2.filter2D`, `cv2.morphologyEx`, `cv2.getStructuringElement`, `cv2.distanceTransform`, `cv2.GaussianBlur` |
| **SciPy** | 梯度计算、高斯平滑、连通域标记 | `scipy.ndimage.sobel`, `scipy.ndimage.gaussian_filter`, `scipy.ndimage.label`, `scipy.stats.norm` |
| **scikit-image** | 形态学骨架化 | `skimage.morphology.skeletonize` |
| **concurrent.futures** | 多进程批处理 | `ProcessPoolExecutor` |

---

## 安装

```bash
# 使用 uv (推荐)
uv sync

# 或 pip
pip install numpy opencv-python-headless scipy scikit-image
```

## 使用方法

```bash
# 运行完整纤维识别管线 (融合所有方法)
python main.py

# 生成高噪声弱纤维训练数据
python main.py --generate
python main.py --generate --num 500

# 指定检测方法
python main.py --method bandpass    # 仅频域带通滤波
python main.py --method gabor       # 仅 Gabor 方向滤波
python main.py --method tophat      # 仅 Top-Hat 形态学
python main.py --method hessian     # 仅 Hessian 脊线检测
python main.py --method all         # 融合所有方法 (默认)

# 指定 I/O 目录
python main.py --input ./images --output ./out

# 并行处理
python main.py --workers 8
```

## 参数配置

所有可调参数集中在 `main.py` 文件头部的全局变量中（第 52-134 行），包括：

- **数据生成参数：** `GEN_SIZE_MODE`, `GEN_NOISE_INTENSITY` 等
- **频域带通参数：** `BP_LOW_CUT`, `BP_HIGH_CUT`, `BP_ORDER`
- **Gabor 参数：** `GABOR_THETAS`, `GABOR_SIGMA`, `GABOR_LAMBDA`, `GABOR_GAMMA`
- **Top-Hat 参数：** `TOPHAT_KERNEL_W`, `TOPHAT_KERNEL_H`
- **Hessian 参数：** `HESSIAN_SCALES`, `LINE_BETA`
- **融合权重：** `WEIGHT_BANDPASS`, `WEIGHT_GABOR`, `WEIGHT_TOPHAT`, `WEIGHT_HESSIAN`
- **后处理参数：** `BINARIZE_METHOD`, `MIN_COMPONENT_AREA`, `ELONGATION_MIN` 等

---

## 项目结构

```
ossifyingFibroma/
├── main.py          # 主程序 — 数据生成 + 纤维检测管线
├── README.md        # 本文档
├── pyproject.toml   # 项目配置
├── uv.lock          # 依赖锁定
├── .python-version  # Python 版本
├── modules/
│   ├── __init__.py
│   └── processing.py
└── .venv/           # 虚拟环境
```

---

## License

MIT
