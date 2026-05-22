# Fingerprint Restoration

指纹图像增强与修复工具，针对低质量指纹图像进行预处理，提升脊线清晰度与可识别性。

## 项目概述

本项目的核心目标是对模糊、噪声严重的指纹图像进行增强处理，通过一系列图像处理与计算机视觉算法，恢复清晰的指纹脊线结构，最终输出二值化增强图像，适用于指纹识别、特征提取等下游任务。

处理流水线如下：

```
输入图像 → 指纹分割 (GMFS) → 方向场估计 (GBFOE) → 频率估计 (XSFFE) 
    → Gabor 滤波增强 (GBFEN) → Otsu 二值化 → 输出图像
```

---

## 数学模型与方法

### 1. GMFS —— 基于梯度幅值的指纹分割

**全称：Gradient Magnitude based Fingerprint Segmentation**

该方法用于将指纹区域与背景分离。其核心思想是利用指纹脊线的高对比度特性——脊线区域具有较高的梯度幅值，而背景区域则相对平滑。

**数学原理：**

1. **梯度幅值计算**：使用 Sobel 算子计算图像在 x 和 y 方向的梯度，进而得到梯度幅值：
   $$
   G_x = \frac{\partial I}{\partial x},\quad G_y = \frac{\partial I}{\partial y},\quad m = \sqrt{G_x^2 + G_y^2}
   $$

2. **高斯平滑**：对梯度幅值图进行高斯滤波，抑制噪声干扰：
   $$
   m_a = G_\sigma * m
   $$

3. **自适应阈值分割**：基于梯度幅值的百分位数确定阈值，生成二值掩膜：
   $$
   \text{mask}(x,y) = \begin{cases} 1, & m_a(x,y) > P_{95}(m) \times 0.2 \\ 0, & \text{else} \end{cases}
   $$

4. **形态学后处理**：通过闭运算（Closing）填充孔洞、开运算（Opening）去除噪点，最后通过连通分量分析保留最大连通区域。

---

### 2. GBFOE —— 基于梯度的指纹方向场估计

**全称：Gradient-Based Fingerprint Orientation Estimation**

该方法用于估计指纹图像中每个像素处脊线的局部方向。指纹的方向场是指纹图像最重要的全局特征之一。

**数学原理：**

1. **对比度拉伸**：通过百分位截断增强图像对比度：
   $$
   I_{\text{norm}} = \text{clip}\left(\frac{I - P_{19}}{P_{81} - P_{19}} \times 255,\ 0,\ 255\right)
   $$

2. **梯度计算**：使用 Sobel 算子计算图像的梯度：
   $$
   G_x = \text{Sobel}_x(I),\quad G_y = \text{Sobel}_y(I)
   $$

3. **梯度协方差矩阵**：计算梯度的二阶矩：
   $$
   G_{xx} = G_x^2,\quad G_{yy} = G_y^2,\quad G_{xy} = -2G_x G_y
   $$

4. **高斯加权聚合**：对梯度矩进行高斯加权平均，得到局部结构张量：
   $$
   \bar{G}_{xx} = G_\sigma * G_{xx},\quad \bar{G}_{yy} = G_\sigma * G_{yy},\quad \bar{G}_{xy} = G_\sigma * G_{xy}
   $$

5. **方向场计算**：利用结构张量中两个特征值对应的特征向量方向，估计脊线方向：
   $$
   \theta = \frac{1}{2} \arctan\left(\frac{\bar{G}_{xy}}{\bar{G}_{xx} - \bar{G}_{yy}}\right)
   $$
   注意，指纹脊线方向与梯度方向垂直，因此需要对结果进行转换。

6. **方向场可靠性**（强度/一致性）：通过结构张量的特征值衡量方向估计的可信度：
   $$
   s = \frac{\sqrt{(\bar{G}_{xx} - \bar{G}_{yy})^2 + \bar{G}_{xy}^2}}{\bar{G}_{xx} + \bar{G}_{yy}}
   $$

---

### 3. XSFFE —— 基于 X 签名的指纹脊线频率估计

**全称：X-Signature Fingerprint Frequency Estimation**

该方法用于估计指纹图像中脊线的局部频率（即相邻脊线之间的像素距离）。脊线频率是 Gabor 滤波器设计中不可或缺的参数。

**数学原理：**

1. **预处理**：对图像进行中值滤波和高斯滤波，减少噪声。

2. **方向对齐采样**：在每个采样点，以脊线方向为基准旋转图像块，使脊线方向垂直向上：
   $$
   \text{region} = \text{warpAffine}(I, R_{-\theta}, (w, h))
   $$

3. **X 签名投影**：沿水平方向（即垂直于脊线的方向）对旋转后的图像块进行求和投影：
   $$
   xs(y) = \sum_{x=1}^{W} \text{region}(y, x)
   $$
   该投影信号具有周期性的波峰和波谷，对应于脊线和谷线。

4. **峰值/谷值检测**：寻找 X 签名的一阶导数为零且二阶导数为负（峰值）或正（谷值）的位置：
   $$
   \text{peaks} = \{y\ |\ xs'(y)=0,\ xs''(y)<0\}
   $$
   $$
   \text{valleys} = \{y\ |\ xs'(y)=0,\ xs''(y)>0\}
   $$

5. **周期估计**：计算相邻峰值/谷值之间的距离，取中位数作为该块的脊线周期：
   $$
   d = \{p_{i+1} - p_i\ |\ p \in \text{peaks} \cup \text{valleys}\}
   $$
   $$
   \text{period} = \text{median}(d)
   $$

6. **缺失区域修复**：对未能成功估计周期的区域，使用 Navier-Stokes 图像修复（inpainting）算法进行插值填充，最后进行高斯平滑。

---

### 4. GBFEN —— 基于 Gabor 滤波器的指纹增强

**全称：Gabor-Based Fingerprint ENhancement**

这是整个系统的核心增强模块，使用 Gabor 滤波器组对指纹图像进行上下文相关的方向滤波。

**数学原理：**

1. **Gabor 滤波器核**：Gabor 滤波器是一个由高斯包络调制的正弦平面波，具有优异的空间-频率局部化特性：
   $$
   G(x, y; \lambda, \theta, \psi, \sigma, \gamma) = 
   \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right)
   \cos\left(2\pi\frac{x'}{\lambda} + \psi\right)
   $$
   其中：
   - $x' = x\cos\theta + y\sin\theta$
   - $y' = -x\sin\theta + y\cos\theta$
   - $\lambda$：波长（对应脊线周期）
   - $\theta$：方向角
   - $\psi$：相位偏移（取 0）
   - $\sigma$：高斯包络标准差（$\sigma = 5\lambda/12$）
   - $\gamma$：空间纵横比（取 1.0）

2. **Gabor 滤波器组构建**：构建覆盖多个方向和多个周期的滤波器组：
   - 方向数：16 个（$0, \pi/16, 2\pi/16, \dots, 15\pi/16$）
   - 周期数：9 个（从 5 到 20 像素等间隔采样）
   - 总共：$16 \times 9 = 144$ 个 Gabor 滤波器核

3. **上下文相关滤波**（Contextual Convolution）：对于每个像素，根据其局部方向场和频率估计值，从滤波器组中选择最匹配的 Gabor 核进行滤波：
   $$
   I_{\text{enhanced}}(x,y) = \text{filter2D}(I_{\text{invert}}, G_{\theta(x,y), \lambda(x,y)})
   $$
   其中 $I_{\text{invert}}$ 是反转图像（白色脊线、黑色背景），$\theta(x,y)$ 和 $\lambda(x,y)$ 分别是该像素的局部方向和脊线周期。

---

### 5. Fingerprint Enhancer —— 方向 Gabor 滤波器组增强

`fingerprint_enhancer` 库实现了另一种基于 Gabor 滤波器的指纹增强方案（参考 Peter Kovesi 的工作和 Hong et al. 的经典论文）。

**处理流程的数学建模：**

1. **图像归一化**：将脊线区域的灰度值归一化为零均值、单位标准差：
   $$
   I_{\text{norm}}(x,y) = \frac{I(x,y) - \mu}{\sigma}
   $$
   其中 $\mu$ 和 $\sigma$ 是脊线区域像素的均值和标准差。

2. **脊线分割**：将图像划分为 $16 \times 16$ 的块，对每个块计算标准差：
   $$
   \text{mask}(i,j) = \begin{cases} 1, & \sigma_{ij} > T \\ 0, & \text{else} \end{cases}
   $$

3. **方向估计**：使用梯度协方差方法，计算局部结构张量。

4. **频率估计**：沿垂直于脊线的方向投影，进行傅里叶分析，估计脊线频率。

5. **定向 Gabor 滤波**：构造与局部方向和频率匹配的 Gabor 滤波器，进行滤波增强。

---

### 6. Otsu 大津二值化

在增强完成后，使用 Otsu 算法对灰度增强图像进行自适应二值化，将图像转换为黑白二值图像。

**数学原理：**

Otsu 算法通过最大化类间方差（Between-class Variance）来确定最优阈值 $T^*$：
$$
T^* = \arg\max_{T} \sigma_b^2(T)
$$
其中类间方差定义为：
$$
\sigma_b^2 = \omega_0 \omega_1 (\mu_0 - \mu_1)^2
$$
- $\omega_0, \omega_1$：前景和背景像素的比例
- $\mu_0, \mu_1$：前景和背景的灰度均值

---

## 项目结构

```
├── main.py               # 主处理脚本
├── pyproject.toml        # 项目配置与依赖
├── .venv/                # Python 虚拟环境
├── ../2Rec/              # 输入指纹图像目录
└── ../results/           # 增强结果输出目录
```

## 依赖项

- **Python ≥ 3.12**
- **OpenCV**：图像处理与计算机视觉基础库
- **NumPy**：数值计算
- **pyfing**：指纹处理库，提供 GMFS、GBFOE、XSFFE、GBFEN 等方法
- **fingerprint-enhancer**：指纹增强库，提供方向 Gabor 滤波器组实现
- **TensorFlow / Keras**：深度学习框架（部分算法需要）

## 参考文献

1. Hong, L., Wan, Y., & Jain, A. K. (1998). Fingerprint image enhancement: Algorithm and performance evaluation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(8), 777–789.
2. Kovesi, P. (2005). MATLAB and Octave functions for computer vision and image processing. School of Computer Science & Software Engineering, The University of Western Australia.
3. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
4. Gabor, D. (1946). Theory of communication. *Journal of the Institution of Electrical Engineers*, 93(26), 429–457.
