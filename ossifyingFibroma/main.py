"""
================================================================================
显微镜灰度图 — 高噪声背景下神经元纤维识别系统
Neuron Fiber Detection under Heavy Noise in Microscopy
================================================================================

核心痛点:
  1. 背景噪声大 (显微成像噪声 + 光照不均 + 组织纹理)
  2. 纤维亮度低于噪声 (极弱神经元纤维, 灰度仅 20-40)

传统"全局二值化"或"高斯模糊+差分"在此场景下完全失效。

────────────────────────────────────────────────────────────────────────────────
解决方案 — 四大数学模型 + 数据生成 + 后处理管线
────────────────────────────────────────────────────────────────────────────────

一、数据生成模块 (模拟真实显微组织纹理)
  ┌─ Butterworth 带通纹理 (频域滤波法)
  │   数学模型: N(0,1) 高斯白噪声 → FFT → Butterworth 带通滤波器 H(ω) 相乘
  │             → IFFT → 标准差归一化
  │   滤波器: H(ω) = 1/(1+(ω/ω_high)^(2n)) × (1 - 1/(1+(ω/ω_low)^(2n)))
  │   工具: numpy.fft.fft2 / ifft2, numpy.ogrid
  │
  ├─ 光照不均模拟
  │   数学模型: 多个随机高斯斑点叠加 (∑ A·exp(-||x-μ||²/2σ²))
  │   工具: OpenCV GaussianBlur
  │
  └─ 显微传感噪声
      数学模型: 泊松噪声 P(λ·I) / λ + 高斯白噪声 N(0, σ²)
      工具: numpy.random.poisson / normal

二、传统CV纤维检测方法 (四大特征提取模型)
  ┌─ 2.1 频域带通滤波 (Butterworth Bandpass)
  │   数学模型: FFT → Butterworth 带通滤波器 H(ω) = HP(ω)·LP(ω)
  │             HP(ω) = 1/(1+(ω_low/ω)^(2n)), LP(ω) = 1/(1+(ω/ω_high)^(2n))
  │   作用: 滤除光照不均(极低频)和椒盐噪声(极高频), 保留纤维中频方向性
  │   工具: numpy.fft, numpy.ogrid
  │
  ├─ 2.2 Gabor 方向滤波器组
  │   数学模型: Gabor 核 = 高斯包络 × 正弦载波
  │             g(x,y) = exp(-(x'²+γ²y'²)/2σ²) × cos(2πx'/λ + ψ)
  │             其中 x' = xcosθ + ysinθ, y' = -xsinθ + ycosθ
  │   作用: 多方向(6方向)卷积 → 逐像素取最大值, 匹配任意方向纤维
  │   工具: cv2.getGaborKernel, cv2.filter2D
  │
  ├─ 2.3 Top-Hat 形态学变换
  │   数学模型: TopHat(A) = A - (A ⊖ B) ⊕ B   (开运算)
  │             BlackHat(A) = (A ⊕ B) ⊖ B - A  (闭运算)
  │             多角度旋转结构元素 → 逐像素取最大响应
  │   作用: 提取局部邻域内"最亮/最暗"的细长纤维结构
  │   工具: cv2.morphologyEx(MORPH_TOPHAT / MORPH_BLACKHAT),
  │         cv2.getStructuringElement(MORPH_ELLIPSE), cv2.warpAffine
  │
  └─ 2.4 多尺度 Hessian 脊线检测 (Frangi 滤波)
      数学模型: Hessian 矩阵 H = [[Ixx, Ixy], [Ixy, Iyy]]
               特征值 λ₁ ≥ λ₂, 脊线判定:
                 - 亮纤维暗背景: strength = max(-λ₂, 0)
                 - Frangi 线状因子: exp(-R²/2β²), R = |λ₁|/|λ₂|
               多尺度融合: 逐像素取各 σ ∈ [1,2,3,4,5] 的最大响应
      作用: 利用二阶偏导特征值区分纤维状(λ₂≪0≈λ₁) vs 团状(λ₁≈λ₂)
      工具: scipy.ndimage.sobel, scipy.ndimage.gaussian_filter

三、多方法融合与后处理
  ┌─ 加权融合: fused = Σ w_i · R_i, 权重归一化到 [0,1]
  ├─ 二值化: Otsu / 分位数阈值 (mean + σ·Φ⁻¹(1-α))
  ├─ 形态学清理: 闭运算(连接缝隙) → 开运算(去除噪点)
  ├─ 伸长率过滤: 协方差矩阵 Σ = [[σxx, σxy], [σxy, σyy]]
  │             伸长率 e = √(λ₁/λ₂), λ₁≥λ₂ 为 Σ 的特征值
  │             圆形 ≈ 1.0, 纤维 ≫ 1.0, 保留 e ≥ 1.5 的组件
  ├─ PCA 趋势线降噪:
  │   数学模型: 主组件像素集 → 协方差矩阵 → 特征分解
  │            第一主成分方向 = 趋势线方向
  │            自适应距离阈值: P95(残差) × 5.0
  │   工具: numpy.cov, numpy.linalg.eigh
  ├─ 骨架化: 距离变换 (cv2.distanceTransform) + 形态学骨架 (skimage.skeletonize)
  │          恢复原始宽度: 骨架点 → 以距离变换值为半径画圆盘
  └─ 边缘平滑: 高斯模糊 + 重新二值化

────────────────────────────────────────────────────────────────────────────────
技术栈
────────────────────────────────────────────────────────────────────────────────
  - Python 3.12+
  - NumPy          — 矩阵运算、FFT、PCA 特征分解、统计分布
  - OpenCV (cv2)   — Gabor 核、形态学变换、距离变换、图像 I/O
  - SciPy          — sobel 梯度、gaussian_filter、连通域标记(label)
  - scikit-image   — 形态学骨架化 (skeletonize)
  - concurrent.futures — 多进程批处理

Usage:
  python main.py                          # 运行完整纤维识别管线 (融合所有方法)
  python main.py --generate               # 生成高噪声弱纤维训练数据
  python main.py --method bandpass        # 仅频域带通滤波
  python main.py --method gabor           # 仅 Gabor 方向滤波
  python main.py --method tophat          # 仅 Top-Hat 形态学
  python main.py --method hessian         # 仅 Hessian 脊线检测
  python main.py --method all             # 融合所有方法 (纤维识别默认)
  python main.py --input ./images --output ./out  # 指定 I/O 目录
  python main.py --workers 4              # 4 进程并行
"""

from __future__ import annotations

import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import label, gaussian_filter, sobel
from scipy.stats import norm

# ═══════════════════════════════════════════════════════════════
# 全局参数 — 集中管理所有可调参数（神经元纤维识别）
# ═══════════════════════════════════════════════════════════════

# ---- 数据生成参数 ----
GEN_SIZE_MODE     = "random"      # "fixed"=固定512x512, "random"=随机尺寸
GEN_WIDTH         = 512           # 生成图像宽度 (size_mode=fixed 时)
GEN_HEIGHT        = 512           # 生成图像高度 (size_mode=fixed 时)
GEN_NOISE_INTENSITY = 8           # 相机传感噪声 σ (纹理是主方差源, 传感噪声仅微弱叠加)
GEN_SCRATCH_BRIGHTNESS_LO = 25    # 亮纤维灰度下限
GEN_SCRATCH_BRIGHTNESS_HI = 55    # 亮纤维灰度上限
GEN_SCRATCH_DARKNESS_LO  = 5      # 暗纤维灰度下限
GEN_SCRATCH_DARKNESS_HI  = 20     # 暗纤维灰度上限
GEN_SCRATCH_THICKNESS_MIN = 1     # 纤维线宽下限 (px)
GEN_SCRATCH_THICKNESS_MAX = 3     # 纤维线宽上限 (px)
GEN_NUM_SCRATCHES_MIN = 1         # 最少纤维数
GEN_NUM_SCRATCHES_MAX = 3         # 最多纤维数
GEN_BG_MAX        = 30            # 背景亮斑上限
GEN_BG_BLUR       = 101           # 背景平滑核

# ---- 频域带通滤波参数 ----
BP_LOW_CUT      = 0.02          # 低频截止 (归一化频率, 0~0.5, 滤除光照不均)
BP_HIGH_CUT     = 0.35          # 高频截止 (滤除纯随机椒盐噪声)
BP_ORDER        = 2             # Butterworth 滤波器阶数

# ---- Gabor 方向滤波参数 ----
GABOR_THETAS    = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]  # 6个方向
GABOR_KERNEL_SIZE = 21          # Gabor 核大小 (奇数)
GABOR_SIGMA     = 4.0           # 高斯包络 σ
GABOR_LAMBDA    = 10.0          # 正弦波长
GABOR_GAMMA     = 0.5           # 空间长宽比 (越小越细长)
GABOR_PSI       = 0             # 相位偏移

# ---- Top-Hat 形态学参数 ----
TOPHAT_KERNEL_W = 31            # 结构元素宽度 (长轴, 对齐纤维方向)
TOPHAT_KERNEL_H = 3             # 结构元素高度 (短轴, 略宽于纤维线宽)
TOPHAT_METHOD   = "tophat"      # "tophat"=亮线, "blackhat"=暗线

# ---- Hessian 脊线检测参数 (保留) ----
HESSIAN_SCALES  = [1.0, 2.0, 3.0, 4.0, 5.0]
HESSIAN_MODE    = "bright"      # "bright"=亮纤维暗背景, "dark"=暗纤维亮背景
LINE_BETA       = 0.5           # Frangi β: 团状惩罚敏感度

# ---- 融合权重 ----
WEIGHT_BANDPASS = 0.25          # 频域带通结果权重
WEIGHT_GABOR    = 0.35          # Gabor 方向滤波权重
WEIGHT_TOPHAT   = 0.20          # Top-Hat 权重
WEIGHT_HESSIAN  = 0.20          # Hessian 脊线权重

# ---- 二值化 ----
BINARIZE_METHOD   = "mean"      # "otsu" | "fixed" | "mean"
BINARIZE_RATIO    = 0.15        # "mean" 方法的上分位数
BINARIZE_THRESHOLD = 0.06       # "fixed" 方法的相对阈值

# ---- 形态学清理 ----
CLOSE_KERNEL = 5                # 闭运算核大小
CLOSE_ITER   = 2                # 闭运算迭代次数
OPEN_KERNEL  = 3                # 开运算核大小
OPEN_ITER    = 1                # 开运算迭代次数

# ---- 伸长率过滤 ----
MIN_COMPONENT_AREA = 15         # 最小连通域面积
ELONGATION_MIN     = 1.5        # 最小伸长率 √(λ₁/λ₂)
TOP_K              = 100        # 最多保留组件数

# ---- 趋势线降噪 (动态自适应: 找到主纤维趋势路径, 保留路径上的块) ----
DENOISE_ENABLE     = True       # 是否启用趋势线降噪

# ---- 骨架化: 将离散纤维片段聚合成连续线 (线可拐弯、可分裂) ----
GATHER_ENABLE        = True       # 是否启用骨架化聚线
GATHER_CLOSE_RATIO   = 0.02       # 闭运算核大小 = ratio * min(h,w)
MIN_SKELETON_LENGTH  = 10         # 最短骨架保留长度 (px)


# ---- 最终平滑 ----
SMOOTH_SIGMA    = 0.8           # 高斯平滑 σ
SMOOTH_THRESHOLD = 128          # 平滑后二值化阈值

# ---- I/O ----
INPUT_DIR   = "../3Seg"
OUTPUT_DIR  = "../results"
GEN_OUTPUT_DIR = "../generated"  # 数据生成输出目录
NUM_WORKERS = 8


# ╔══════════════════════════════════════════════════════════════╗
# ║  一、数据生成模块 — 模拟真实显微组织纹理                      ║
# ╚══════════════════════════════════════════════════════════════╝
#
# 神经元纤维灰度图像统计特性:
#   - 亮度均值 38.6 ± 6.8, 标准差 26.6 ± 2.9
#   - 相邻像素差 P95 ≈ 33, 局部标准差 ≈ 19
#   - 梯度幅值 P95 ≈ 187, Laplacian 方差 ≈ 1741
#   - Canny 边缘占比 ≈ 36%, Hough 线 ≈ 1242 条
#   - 角度熵 ≈ 0.993 (各向同性纹理), 主导方向强度 ≈ 0.04
#   - FFT 高频/中频能量比 ≈ 0.955
#
# 核心思路:
#   显微组织噪声不是传感器白噪声，而是组织本身的纹理。
#   我们用频域滤波法生成多尺度随机纹理 (Bandpass Noise Texture)，
#   再叠加大尺度光照不均，最后画入被纹理掩盖的微弱神经元纤维。



def _bandpass_texture(
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    low_cut: float = 0.015,    # 低频截止 (归一化频率, 校准自3Seg)
    high_cut: float = 0.22,    # 高频截止
    order: float = 1.8,        # 过渡带陡峭度
    target_std: float = 15.0,  # 纹理内部标准差 (校准自3Seg统计)
) -> np.ndarray:
    """生成频域带通纹理 —— 模拟显微组织粗糙纹理.

    在频域中构造 Butterworth 带通滤波器，对高斯白噪声进行滤波，
    得到能量集中在 [low_cut, high_cut] 频段的各向同性纹理。
    这种纹理的统计特性 (局部标准差、梯度幅值、Canny 边缘占比)
    与真实显微图像高度一致。

    Parameters
    ----------
    height, width : int
        纹理尺寸.
    rng : np.random.Generator
        随机数生成器.
    low_cut : float
        低频截止 (DC 附近, 用于去除大面积不均匀).
    high_cut : float
        高频截止 (滤除纯随机像素噪声).
    order : float
        Butterworth 阶数.
    target_std : float
        目标标准差 (匹配真实数据 26.6 ± 2.9).

    Returns
    -------
    texture : np.ndarray, float64
        带通纹理, 值域约 [-1, 1].
    """
    # 高斯白噪声
    noise = rng.normal(0, 1, (height, width))

    # FFT
    F = fft2(noise)
    Fshift = fftshift(F)

    # 构造 Butterworth 带通滤波器
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_r = np.sqrt(cy**2 + cx**2)
    dist_norm = dist / max_r  # [0, 1]

    eps = 1e-9
    low_pass = 1.0 / (1.0 + (dist_norm / (high_cut + eps))**(2 * order))
    high_pass = 1.0 - 1.0 / (1.0 + (dist_norm / (low_cut + eps))**(2 * order))
    bandpass = low_pass * high_pass

    # 应用滤波器
    F_filtered = Fshift * bandpass
    F_ifft = ifftshift(F_filtered)
    texture = np.real(ifft2(F_ifft))

    # 归一化到目标标准差
    curr_std = texture.std()
    if curr_std > 1e-9:
        texture = texture * (target_std / curr_std)

    return texture


def _add_illumination_variation(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    num_blobs: int = 3,
    blob_scale_range: tuple[float, float] = (0.15, 0.40),
    intensity_range: tuple[float, float] = (5, 20),
) -> np.ndarray:
    """叠加模拟光照不均的大尺度亮度变化.

    用大尺寸高斯斑块模拟光源不均匀或组织区域性反光。
    """
    h, w = img.shape
    illumination = np.zeros((h, w), dtype=np.float64)

    n = rng.integers(1, num_blobs + 1)
    for _ in range(n):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        sigma = rng.uniform(*blob_scale_range) * min(h, w)
        amp = rng.uniform(*intensity_range)
        # 生成高斯斑块
        y, x = np.ogrid[:h, :w]
        blob = amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        illumination += blob

    return img + illumination


def _random_size(rng: np.random.Generator) -> tuple[int, int]:
    """从真实尺寸分布中随机采样图像尺寸."""
    if GEN_SIZE_MODE == "fixed":
        return GEN_WIDTH, GEN_HEIGHT
    w = int(rng.integers(400, 800))
    h = int(rng.integers(400, 800))
    return w, h



def generate_weak_scratch(
    width: int | None = None,
    height: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    num_scratches: int | None = None,
    scratch_brightness_range: tuple[int, int] = (25, 55),
    scratch_darkness_range: tuple[int, int] = (5, 20),
    thickness_range: tuple[int, int] = (1, 4),
    noise_intensity: float = GEN_NOISE_INTENSITY,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一张「真实显微纹理 + 极弱神经元纤维」仿真图及其真值 mask.

    关键改进:
      1. 用频域带通纹理替代简单的 Gaussian blur + Poisson 噪声,
         精准匹配真实显微组织的纹理统计特性.
      2. 大尺度光照不均用高斯斑块模拟, 而非整图模糊的随机噪声.
      3. 纤维亮度/暗度相对于局部纹理而言是微弱差异,
         但方向连续性好, 这正是算法的核心挑战.
      4. 支持多条纤维 (亮线 + 暗线混合).

    Parameters
    ----------
    width, height : int or None
        图像尺寸. None 则从真实尺寸分布中随机采样.
    rng : np.random.Generator or None
        随机数生成器.
    num_scratches : int or None
        纤维数量. None 则随机 1~3 条.
    scratch_brightness_range : tuple[int, int]
        亮纤维灰度值范围 (相对纹理偏高).
    scratch_darkness_range : tuple[int, int]
        暗纤维灰度值范围 (相对纹理偏低).
    thickness_range : tuple[int, int]
        纤维线宽范围 (px).
    noise_intensity : float
        高斯噪声 σ (叠加在纹理之上, 模拟相机传感噪声).
    Returns
    -------
    noisy_img : np.ndarray, uint8
        含噪声 + 纹理的最终图像.
    gt_mask : np.ndarray, uint8
        纤维真值二值 mask (0/255). 注意: mask 不含纹理, 仅为纤维.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 尺寸
    if width is None or height is None:
        width, height = _random_size(rng)

    # ---- 步骤1: 生成带通纹理 (模拟显微组织粗糙度) ----
    tex_std = rng.uniform(13, 17)  # 纹理内部标准差
    texture = _bandpass_texture(
        height, width, rng,
        low_cut=rng.uniform(0.012, 0.018),
        high_cut=rng.uniform(0.20, 0.24),
        order=rng.uniform(1.6, 2.0),
        target_std=tex_std,
    )

    # ---- 步骤2: 叠加光照不均 ----
    texture = _add_illumination_variation(
        texture, rng,
        num_blobs=4,
        blob_scale_range=(0.12, 0.35),
        intensity_range=(5, 22),
    )

    # ---- 步骤3: 将纹理映射到 [0, 255] ----
    # 目标: 均值 ≈ 39 (匹配真实均值), 标准差 ≈ 22-28
    target_mean = rng.uniform(35, 45)
    tex_mean = texture.mean()
    tex_curr_std = texture.std()
    if tex_curr_std < 1e-9:
        tex_curr_std = 1.0

    # 将纹理标准化后缩放到目标标准差，平移到目标均值
    texture_norm = (texture - tex_mean) / tex_curr_std
    target_std = rng.uniform(20, 28)  # 匹配真实数据 std=26.6±2.9
    img_float = texture_norm * target_std + target_mean

    # 裁剪到 [0, 255]
    img_float = np.clip(img_float, 0, 255)
    img_base = img_float.astype(np.uint8)

    # ---- 步骤4: 画神经元纤维 ----
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    if num_scratches is None:
                out_path = output_dir / f"3Seg_{img_path.stem}.png"  # 1~3 条
  # 1~3 条
    assert num_scratches is not None

    for _ in range(num_scratches):
        # 随机起止点
        x1 = int(rng.integers(0, width // 2))
        y1 = int(rng.integers(0, height // 2))
        x2 = int(rng.integers(width // 2, width))
        y2 = int(rng.integers(height // 2, height))

        thickness = int(rng.integers(*thickness_range))

        # 50% 概率为亮纤维, 50% 为暗纤维
        if rng.random() < 0.5:
            # 亮纤维: 灰度略高于周围纹理
            brightness = int(rng.integers(*scratch_brightness_range))
            cv2.line(img_base, (x1, y1), (x2, y2), brightness, thickness, lineType=cv2.LINE_AA)
        else:
            # 暗纤维: 灰度低于周围纹理
            darkness = int(rng.integers(*scratch_darkness_range))
            # 画暗线: 在 mask 上标记, 之后用 addWeighted 降低该区域亮度
            temp_dark = np.zeros((height, width), dtype=np.uint8)
            cv2.line(temp_dark, (x1, y1), (x2, y2), 255, thickness, lineType=cv2.LINE_AA)
            dark_factor = darkness / 255.0
            img_float2 = img_base.astype(np.float64)
            img_float2[temp_dark > 0] = img_float2[temp_dark > 0] * (1.0 - dark_factor) + darkness * dark_factor
            img_base = np.clip(img_float2, 0, 255).astype(np.uint8)

        # 更新真值 mask
        cv2.line(gt_mask, (x1, y1), (x2, y2), 255, thickness, lineType=cv2.LINE_AA)

    # ---- 步骤5: 叠加显微传感噪声 (泊松 + 高斯) ----
    noisy_img = add_realistic_noise(img_base, intensity=noise_intensity)

    return noisy_img, gt_mask


def add_realistic_noise(img: np.ndarray, intensity: float = 8) -> np.ndarray:
    """模拟真实显微传感噪声: 泊松噪声 + 高斯白噪声.

    注意: 新版数据生成中纹理已占据主要方差, 传感噪声仅作为
    传感器层面的微弱叠加。因此默认 intensity 降低到 8 (旧版 30-40),
    否则会过度掩盖纹理结构。

    Parameters
    ----------
    img : np.ndarray
        输入灰度图像 (uint8).
    intensity : float
        高斯噪声的标准差 σ. 默认 8 (匹配真实噪声水平).

    Returns
    -------
    noisy : np.ndarray
        叠加噪声后的图像 (uint8).
    """
    img_f = img.astype(np.float64)

    # 泊松噪声
    vals = 2 ** np.ceil(np.log2(max(len(np.unique(img)), 2)))
    poisson_noisy = np.random.poisson(img_f * vals) / float(vals)

    # 高斯噪声
    gauss = np.random.normal(0, intensity, img.shape)

    noisy = poisson_noisy + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def generate_dataset(
    num_samples: int = 100,
    output_dir: str | Path = GEN_OUTPUT_DIR,
    *,
    seed: int | None = None,
) -> None:
    """批量生成高噪声弱纤维数据集 (图像 + 真值 mask).

    Parameters
    ----------
    num_samples : int
        生成样本数.
    output_dir : str | Path
        输出目录, 将创建 images/ 和 masks/ 子目录.
    seed : int or None
        随机种子 (可复现).
    """
    out = Path(output_dir)
    img_dir = out / "images"
    msk_dir = out / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"生成 {num_samples} 张真实纹理+弱纤维仿真图像...")
    for i in range(num_samples):
        noisy, gt = generate_weak_scratch(rng=rng)
        cv2.imwrite(str(img_dir / f"scratch_{i:04d}.png"), noisy)
        cv2.imwrite(str(msk_dir / f"scratch_{i:04d}_mask.png"), gt)
        if (i + 1) % 50 == 0:
            print(f"  已生成 {i + 1}/{num_samples}")

    print(f"完成 → {img_dir}  +  {msk_dir}")
    # 保存预览图 (固定种子保证预览可比)
    preview_rng = np.random.default_rng(42)
    demo_img, demo_gt = generate_weak_scratch(rng=preview_rng)
    cv2.imwrite(str(out / "preview_noisy.png"), demo_img)
    cv2.imwrite(str(out / "preview_gt.png"), demo_gt)
    print(f"预览图已保存 → {out / 'preview_noisy.png'}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  二、传统CV纤维识别方法                                    ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 2.1 频域带通滤波 (Bandpass Filter) ────────────────────────

def _butterworth_bandpass(
    shape: tuple[int, int],
    low_cut: float = BP_LOW_CUT,
    high_cut: float = BP_HIGH_CUT,
    order: int = BP_ORDER,
) -> np.ndarray:
    """构建 Butterworth 带通频域滤波器.

    滤除:
      - 极低频 (大面积光照不均, 背景渐变)
      - 极高频 (纯随机椒盐噪声)
    保留:
      - 中间频段 (纤维的连续方向性结构)

    Parameters
    ----------
    shape : tuple
        图像 (h, w).
    low_cut, high_cut : float
        归一化截止频率 (0~0.5).
    order : int
        滤波器阶数, 越大过渡越陡.

    Returns
    -------
    filter_mask : np.ndarray
        频域滤波器 (0~1).
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    # 到频谱中心的归一化距离
    dist = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)

    # 高通部分: 抑制低频
    hp = 1.0 / (1.0 + (low_cut / (dist + 1e-12)) ** (2 * order))
    # 低通部分: 抑制高频
    lp = 1.0 / (1.0 + (dist / high_cut) ** (2 * order))

    return hp * lp


def bandpass_filter(
    gray: np.ndarray,
    low_cut: float = BP_LOW_CUT,
    high_cut: float = BP_HIGH_CUT,
    order: int = BP_ORDER,
) -> np.ndarray:
    """频域带通滤波: 提取纤维的中频方向性结构.

    流程:
      1. FFT → 频谱中心化
      2. 乘以 Butterworth 带通滤波器
      3. 逆 FFT → 取模 → 归一化

    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (uint8).
    low_cut, high_cut : float
        归一化截止频率.
    order : int
        滤波器阶数.

    Returns
    -------
    filtered : np.ndarray
        滤波后的响应图 (float64, 0~1).
    """
    h, w = gray.shape
    # FFT
    f = fft2(gray.astype(np.float64))
    fshift = fftshift(f)

    # 构建滤波器
    mask = _butterworth_bandpass((h, w), low_cut, high_cut, order)

    # 应用
    fshift_filtered = fshift * mask

    # 逆变换
    f_ishift = ifftshift(fshift_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 归一化
    vmax = img_back.max()
    if vmax > 0:
        img_back /= vmax

    return img_back


# ── 2.2 Gabor 方向滤波器组 ────────────────────────────────────

def _gabor_kernel(
    ksize: int = GABOR_KERNEL_SIZE,
    sigma: float = GABOR_SIGMA,
    theta: float = 0,
    lambd: float = GABOR_LAMBDA,
    gamma: float = GABOR_GAMMA,
    psi: float = GABOR_PSI,
) -> np.ndarray:
    """生成 Gabor 滤波器核.

    Gabor 核 = 高斯包络 × 正弦载波, 对特定方向/频率的线状结构敏感.
    """
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F,
    )
    return kernel


def gabor_filter_bank(
    gray: np.ndarray,
    thetas: list[float] | None = None,
    ksize: int = GABOR_KERNEL_SIZE,
    sigma: float = GABOR_SIGMA,
    lambd: float = GABOR_LAMBDA,
    gamma: float = GABOR_GAMMA,
) -> np.ndarray:
    """Gabor 方向滤波器组: 多角度卷积 → 逐像素取最大值.

    原理:
      - 每个方向的 Gabor 核对纤维延伸方向产生强响应.
      - 非极大值抑制: 在其他方向响应弱、仅在纤维方向响应强的才是真实纤维.
      - 逐像素取 max 可捕获任意方向的纤维.

    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (uint8).
    thetas : list[float] | None
        方向角列表 (弧度). None 则使用默认 6 方向.
    ksize, sigma, lambd, gamma : Gabor 参数.

    Returns
    -------
    response : np.ndarray
        最大方向响应图 (float64, 0~1).
    """
    if thetas is None:
        thetas = GABOR_THETAS

    gray_f = gray.astype(np.float64)
    h, w = gray_f.shape
    acc = np.zeros((h, w), dtype=np.float64)

    for theta in thetas:
        kernel = _gabor_kernel(ksize=ksize, sigma=sigma, theta=theta,
                                lambd=lambd, gamma=gamma)
        # 卷积
        filtered = cv2.filter2D(gray_f, cv2.CV_64F, kernel)
        # 取模 (响应强度)
        magnitude = np.abs(filtered)
        acc = np.maximum(acc, magnitude)

    # 归一化
    vmax = acc.max()
    if vmax > 0:
        acc /= vmax

    return acc


# ── 2.3 Top-Hat 形态学变换 ────────────────────────────────────

def tophat_transform(
    gray: np.ndarray,
    kernel_w: int = TOPHAT_KERNEL_W,
    kernel_h: int = TOPHAT_KERNEL_H,
    method: str = TOPHAT_METHOD,
) -> np.ndarray:
    """Top-Hat / Black-Hat 形态学变换: 提取局部细长结构.

    使用细长椭圆结构元素:
      - Top-Hat: 原图 - 形态学开运算 → 提取比结构元素细的亮线.
      - Black-Hat: 形态学闭运算 - 原图 → 提取比结构元素细的暗线.

    即使划痕灰度绝对值不高, 只要在局部邻域内是"最亮/最暗"的
    细长结构, 就能被分离出来.

    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (uint8).
    kernel_w : int
        结构元素宽度 (长轴, 沿划痕方向).
    kernel_h : int
        结构元素高度 (短轴, 略大于划痕线宽).
    method : str
        "tophat" (亮线) 或 "blackhat" (暗线).

    Returns
    -------
    response : np.ndarray
        Top-Hat 响应图 (float64, 0~1).
    """
    # 构造细长椭圆结构元素
    if kernel_w % 2 == 0:
        kernel_w += 1
    if kernel_h % 2 == 0:
        kernel_h += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))

    if method == "tophat":
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    elif method == "blackhat":
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    else:
        raise ValueError(f"Unknown tophat method: {method}")

    result_f = result.astype(np.float64)
    vmax = result_f.max()
    if vmax > 0:
        result_f /= vmax

    return result_f


def multi_angle_tophat(
    gray: np.ndarray,
    kernel_w: int = TOPHAT_KERNEL_W,
    kernel_h: int = TOPHAT_KERNEL_H,
    method: str = TOPHAT_METHOD,
    n_angles: int = 6,
) -> np.ndarray:
    """多角度 Top-Hat: 旋转结构元素, 逐像素取最大响应（纤维增强）.

    解决单一方向结构元素对倾斜纤维响应弱的问题.

    Parameters
    ----------
    gray : np.ndarray
        输入灰度图.
    kernel_w, kernel_h : int
        结构元素尺寸.
    method : str
        "tophat" | "blackhat".
    n_angles : int
        旋转角度数 (均匀分布在 0~180°).

    Returns
    -------
    response : np.ndarray
        最大角度响应图 (float64, 0~1).
    """
    h, w = gray.shape
    acc = np.zeros((h, w), dtype=np.float64)

    if kernel_w % 2 == 0:
        kernel_w += 1
    if kernel_h % 2 == 0:
        kernel_h += 1
    base_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))

    for angle in np.linspace(0, 180, n_angles, endpoint=False):
        # 旋转结构元素
        rot_mat = cv2.getRotationMatrix2D((kernel_w / 2, kernel_h / 2), angle, 1.0)
        se_rotated = cv2.warpAffine(
            base_se.astype(np.uint8), rot_mat, (kernel_w, kernel_h),
        )
        se_rotated = (se_rotated > 0.5).astype(np.uint8)

        if method == "tophat":
            resp = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_rotated)
        else:
            resp = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_rotated)

        acc = np.maximum(acc, resp.astype(np.float64))

    vmax = acc.max()
    if vmax > 0:
        acc /= vmax

    return acc


# ── 2.4 多尺度 Hessian 脊线检测 (保留原方案) ──────────────────

def _hessian_eigenvalues(gray: np.ndarray, sigma: float) -> np.ndarray:
    """计算指定尺度 σ 下的 Hessian 矩阵特征值响应.

    Bright ridge on dark bg:  λ₂ ≪ 0, λ₁ ≈ 0 → strength = max(-λ₂, 0)
    Dark valley on bright bg: λ₁ ≫ 0, λ₂ ≈ 0 → strength = max(λ₁, 0)

    Frangi 线状因子 exp(-R²/(2β²)) 惩罚团状区域 (|λ₁| ≈ |λ₂|).
    """
    Ix = sobel(gray, axis=1, mode='reflect')
    Iy = sobel(gray, axis=0, mode='reflect')

    Ix_s = gaussian_filter(Ix.astype(np.float64), sigma, mode='reflect')
    Iy_s = gaussian_filter(Iy.astype(np.float64), sigma, mode='reflect')

    Ixx = sobel(Ix_s, axis=1, mode='reflect')
    Iyy = sobel(Iy_s, axis=0, mode='reflect')
    Ixy = sobel(Ix_s, axis=0, mode='reflect')

    trace = Ixx + Iyy
    det   = Ixx * Iyy - Ixy * Ixy
    disc  = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))

    lam1 = (trace + disc) * 0.5   # λ₁ ≥ λ₂
    lam2 = (trace - disc) * 0.5

    eps = 1e-12
    if HESSIAN_MODE == "bright":
        strength = np.maximum(-lam2, 0.0)
        ratio = np.abs(lam1) / (np.abs(lam2) + eps)
    else:
        strength = np.maximum(lam1, 0.0)
        ratio = np.abs(lam2) / (np.abs(lam1) + eps)

    line_factor = np.exp(-(ratio ** 2) / (2.0 * LINE_BETA ** 2))
    return strength * line_factor


def multi_scale_hessian(gray: np.ndarray) -> np.ndarray:
    """多尺度 Hessian 脊线检测: 逐像素取各尺度最大响应（纤维脊线）."""
    h, w = gray.shape
    acc = np.zeros((h, w), dtype=np.float64)

    for sigma in HESSIAN_SCALES:
        resp = _hessian_eigenvalues(gray, sigma)
        acc = np.maximum(acc, resp)

    vmax = acc.max()
    if vmax > 0:
        acc /= vmax

    return acc


# ╔══════════════════════════════════════════════════════════════╗
# ║  三、多方法融合管线 (综合判定纤维)                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def fuse_responses(
    responses: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """融合多种方法的响应图: 加权叠加 → 归一化（综合纤维判定）.

    Parameters
    ----------
    responses : dict
        {方法名: 响应图 (float64, 0~1)}.
    weights : dict | None
        {方法名: 权重}. None 则使用默认等权重.

    Returns
    -------
    fused : np.ndarray
        融合后的响应图 (float64, 0~1).
    """
    if weights is None:
        weights = {k: 1.0 / len(responses) for k in responses}

    h, w = next(iter(responses.values())).shape
    fused = np.zeros((h, w), dtype=np.float64)

    for name, resp in responses.items():
        w = weights.get(name, 0.0)
        if w > 0 and resp is not None:
            fused += w * resp

    vmax = fused.max()
    if vmax > 0:
        fused /= vmax

    return fused


# ── 二值化 ────────────────────────────────────────────────────

def binarize_response(
    response: np.ndarray,
    method: str = BINARIZE_METHOD,
    ratio: float = BINARIZE_RATIO,
    fixed_threshold: float = BINARIZE_THRESHOLD,
) -> np.ndarray:
    """将连续响应图转为二值 mask（纤维/背景分割）.

    Parameters
    ----------
    response : np.ndarray
        响应图 (float64, 0~1).
    method : str
        "otsu" | "fixed" | "mean".
    ratio : float
        "mean" 方法的上分位数.
    fixed_threshold : float
        "fixed" 方法的相对阈值 (0~1).

    Returns
    -------
    binary : np.ndarray
        二值 mask (uint8, 0/255).
    """
    img = (response * 255.0).clip(0, 255).astype(np.uint8)

    if method == "otsu":
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "fixed":
        t = int(fixed_threshold * 255)
        _, binary = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)

    elif method == "mean":
        positive = img[img > 0]
        if len(positive) == 0:
            return np.zeros_like(img)
        mean_val = float(positive.mean())
        std_val  = float(positive.std()) if len(positive) > 1 else 1.0
        t = mean_val + std_val * norm.ppf(1.0 - ratio)
        t = np.clip(t, 0, 255)
        _, binary = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError(f"Unknown binarize method: {method}")

    return binary


# ── 形态学清理 ────────────────────────────────────────────────

def morph_cleanup(binary: np.ndarray) -> np.ndarray:
    """闭运算连接纤维缝隙 → 开运算去除孤立噪点."""
    cleaned = binary.copy()

    # 闭运算: 连接邻近纤维碎片
    ck = CLOSE_KERNEL
    if ck % 2 == 0:
        ck += 1
    close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    for _ in range(CLOSE_ITER):
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_se)

    # 开运算: 去除盐噪声
    ok = OPEN_KERNEL
    if ok % 2 == 0:
        ok += 1
    open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    for _ in range(OPEN_ITER):
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_se)

    return cleaned


# ── 伸长率过滤 ────────────────────────────────────────────────

def _component_elongation(mask: np.ndarray) -> float:
    """通过协方差矩阵特征值计算连通域伸长率: √(λ_major / λ_minor)（纤维 vs 团状）.

    圆形 ≈ 1.0,  细长纤维 ≫ 1.0.
    """
    ys, xs = np.where(mask)
    n = len(xs)
    if n < 3:
        return 1.0
    cx, cy = xs.mean(), ys.mean()
    a = float(((xs - cx) ** 2).sum()) / n          # σxx
    b = float((2 * (xs - cx) * (ys - cy)).sum()) / n  # 2σxy
    c = float(((ys - cy) ** 2).sum()) / n          # σyy
    disc = np.sqrt(max((a - c) ** 2 + b ** 2, 0.0))
    l1 = (a + c + disc) / 2.0
    l2 = (a + c - disc) / 2.0
    if l2 < 1e-12:
        return 1.0e6
    return float(np.sqrt(l1 / l2))


def filter_elongated(binary: np.ndarray) -> np.ndarray:
    """仅保留细长 (纤维状) 连通域."""
    if not binary.any():
        return binary

    labeled, num_comp = label(binary > 0, structure=np.ones((3, 3), dtype=bool))
    if num_comp == 0:
        return binary

    scored = []
    for lbl in range(1, num_comp + 1):
        mask = labeled == lbl
        area = int(mask.sum())
        if area < MIN_COMPONENT_AREA:
            continue
        elong = _component_elongation(mask)
        if elong < ELONGATION_MIN:
            continue
        scored.append((lbl, area, elong, elong * np.sqrt(area)))

    if not scored:
        # 兜底: 保留面积最大的组件
        areas = [(i, (labeled == i).sum()) for i in range(1, num_comp + 1)]
        if areas:
            areas.sort(key=lambda x: x[1], reverse=True)
            best_lbl = areas[0][0]
            out = np.zeros_like(binary)
            out[labeled == best_lbl] = 255
            return out
        return binary

    scored.sort(key=lambda x: x[3], reverse=True)
    kept = [lbl for lbl, _, _, _ in scored[:TOP_K]]

    out = np.zeros_like(binary)
    for lbl in kept:
        out[labeled == lbl] = 255
    return out


# ── 趋势线降噪 ────────────────────────────────────────────────

def _component_centroid(mask: np.ndarray) -> tuple[float, float]:
    """计算连通域的质心 (cx, cy)."""
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _component_orientation(mask: np.ndarray) -> float:
    """计算连通域主方向 (弧度, 0~π). 通过协方差矩阵特征向量."""
    ys, xs = np.where(mask)
    n = len(xs)
    if n < 3:
        return 0.0
    cx, cy = xs.mean(), ys.mean()
    a = float(((xs - cx) ** 2).sum())
    b = float((2 * (xs - cx) * (ys - cy)).sum())
    c = float(((ys - cy) ** 2).sum())
    # 主方向角度 (弧度)
    theta = 0.5 * np.arctan2(b, a - c)
    return theta


def denoise_outliers(binary: np.ndarray) -> np.ndarray:
    """动态降噪: 找到主纤维趋势路径, 保留路径上的块, 删除路径外的噪声.

    核心思路 (自适应, 确定性, 无随机波动):
      1. 按面积降序排列所有连通域, 取面积最大的组件为主组件候选
      2. 如果最大组件面积占比显著 (>=15%), 认为它连成一片了
         → 用它的所有像素点通过 PCA 拟合趋势线 (确定性)
      3. 如果最大组件不够大, 取前 N 个组件合并像素拟合趋势线
      4. 以趋势线为基准路径, 保留路径附近的组件 (自适应距离阈值)
      5. 远离路径的孤立噪声被清除

    自适应距离阈值:
      - 通过主组件像素点到趋势线的残差分布计算
      - 取 P95 残差 × 系数 作为距离容差 (跟随组件自身宽度变化)
      - 系数 5.0, 下限 10px, 上限 300px

    Parameters
    ----------
    binary : np.ndarray
        二值 mask (uint8, 0/255).

    Returns
    -------
    denoised : np.ndarray
        降噪后的二值 mask.
    """
    if not DENOISE_ENABLE or not binary.any():
        return binary

    labeled, num_comp = label(binary > 0, structure=np.ones((3, 3), dtype=bool))
    if num_comp < 2:
        return binary  # 只有一个组件, 无需降噪

    # 1. 收集所有组件信息，按面积降序
    comps = []
    for lbl in range(1, num_comp + 1):
        mask = (labeled == lbl).astype(np.uint8)
        area = int(mask.sum())
        if area < 3:
            continue
        cx, cy = _component_centroid(mask)
        theta = _component_orientation(mask)
        comps.append({'lbl': lbl, 'mask': mask, 'area': area,
                      'cx': cx, 'cy': cy, 'theta': theta})

    if len(comps) < 2:
        return binary

    comps.sort(key=lambda c: c['area'], reverse=True)
    total_area = sum(c['area'] for c in comps)

    # 2. 找到"连成一片"的主组件
    #    - 如果最大组件面积占比 >= 15%，直接用它的像素拟合趋势线
    #    - 否则取面积占比累加达到 50% 的前 N 个组件
    main_ratio = comps[0]['area'] / max(total_area, 1)
    if main_ratio >= 0.15:
        main_comps = [comps[0]]
    else:
        # 累加直到占 50%
        cum = 0
        main_comps = []
        for c in comps:
            cum += c['area']
            main_comps.append(c)
            if cum / max(total_area, 1) >= 0.50:
                break
        # 至少取前 2 个
        if len(main_comps) < 2 and len(comps) >= 2:
            main_comps = comps[:2]

    # 3. 收集主组件的所有像素点 → 拟合趋势线
    all_pixels = np.zeros((binary.shape[0], binary.shape[1]), dtype=np.uint8)
    for c in main_comps:
        all_pixels[c['mask'] > 0] = 1

    ys, xs = np.where(all_pixels > 0)
    if len(xs) < 5:
        return binary

    fit_pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))

    # PCA 拟合趋势线: 协方差矩阵 → 特征向量 → 主方向 (确定性, 无随机性)
    mean_x = float(np.mean(fit_pts[:, 0]))
    mean_y = float(np.mean(fit_pts[:, 1]))
    centered = fit_pts - np.array([mean_x, mean_y])
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 第一主成分 (最大特征值对应的特征向量) 即为趋势线方向
    principal_vec = eigenvectors[:, 1] if eigenvalues[1] > eigenvalues[0] else eigenvectors[:, 0]
    vx, vy = principal_vec[0], principal_vec[1]
    norm_vec = np.sqrt(vx * vx + vy * vy)
    if norm_vec < 1e-12:
        return binary
    vx /= norm_vec
    vy /= norm_vec

    # 直线: ax + by + c = 0, 其中 (a,b) 是法向量 (垂直于方向向量)
    a, b = vy, -vx
    c_val = -(a * mean_x + b * mean_y)

    # 所有主组件像素点到趋势线的距离 (残差)
    best_residuals = np.abs(fit_pts @ np.array([a, b]) + c_val)

    # 4. 自适应距离阈值: 基于主组件像素到趋势线的 P95 残差
    if best_residuals is not None and len(best_residuals) > 10:
        dist_threshold = max(float(np.percentile(best_residuals, 95)) * 5.0, 10.0)
        dist_threshold = min(dist_threshold, 300.0)
    else:
        dist_threshold = 50.0

    a, b, c = a, b, c_val

    # 5. 用趋势路径过滤所有组件
    kept = []
    for comp in comps:
        cx, cy = comp['cx'], comp['cy']
        dist = abs(a * cx + b * cy + c)
        if dist <= dist_threshold:
            kept.append(comp)

    # 6. 如果过滤太狠（啥都没了），回退到保留主组件 + 合并其附近 1.5× 距离的组件
    if not kept:
        kept = [comps[0]]
        # 以主组件质心为中心，收集附近 1.5× 距离内的组件
        cx0, cy0 = comps[0]['cx'], comps[0]['cy']
        for comp in comps[1:]:
            d = np.sqrt((comp['cx'] - cx0)**2 + (comp['cy'] - cy0)**2)
            if d <= dist_threshold * 1.5:
                kept.append(comp)

    out = np.zeros_like(binary)
    for comp in kept:
        out[comp['mask'] > 0] = 255
    return out


# ── 骨架化: 将离散点聚合成连续线 (线可拐弯、可分裂) ─────────────

def gather_onto_lines(binary: np.ndarray) -> np.ndarray:
    """将降噪后剩余离散纤维点聚合成连续骨架线, 并恢复原始宽度.

    核心思路:
        1. 自适应闭运算连接断裂处 (核大小随图片尺寸变化)
        2. 距离变换 — 记录每个前景像素到背景的距离 (即原始宽度半径)
        3. 形态学骨架化 (skeletonize), 得到单像素宽连续线
        4. 骨架天然支持拐弯 (曲线) 和分裂 (分支/分叉)
        5. 修剪过短的分支 (噪声) — 过滤掉少于 3 个节点的骨架
        6. 对骨架上每个点, 用距离变换值作为半径画圆盘, 恢复原始宽度
           (而不是固定膨胀, 这样粗线和细线都能保持原本的厚度)

    Parameters
    ----------
    binary : np.ndarray
        降噪后的二值 mask (uint8, 0/255).

    Returns
    -------
    skeleton : np.ndarray
        具有原始宽度的骨架化二值 mask (uint8, 0/255).
    """
    if not GATHER_ENABLE or not binary.any():
        return binary

    # 1. 自适应闭运算核大小 (跟随图片尺寸)
    h, w = binary.shape
    close_k = max(3, int(min(h, w) * GATHER_CLOSE_RATIO))
    close_k = close_k if close_k % 2 == 1 else close_k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 2. 距离变换 — 获取每个前景点到背景的距离 (= 原始宽度半径)
    #    cv2.DIST_L2 精确欧氏距离, 值越大表示该处线条越粗
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)

    # 3. 骨架化 → 单像素宽连续线 (可拐弯、可分裂)
    from skimage.morphology import skeletonize
    skel = skeletonize(closed > 0).astype(np.uint8)

    # 4. 修剪过短分支: 过滤掉节点数 < 3 的骨架
    if skel.any():
        labeled, num = label(skel > 0, structure=np.ones((3, 3), dtype=bool))
        for lbl in range(1, num + 1):
            mask = labeled == lbl
            n_nodes = mask.sum()
            # 同时应用 MIN_SKELETON_LENGTH 和最少 3 个节点的约束
            min_nodes = max(MIN_SKELETON_LENGTH, 3)
            if n_nodes < min_nodes:
                skel[mask] = 0

    # 5. 恢复原始宽度: 对骨架上每个点, 以原始半径画圆盘
    if skel.any():
        pts = np.argwhere(skel > 0)
        result = np.zeros_like(skel)
        for y, x in pts:
            r = max(1, int(dist[y, x] + 0.5))
            cv2.circle(result, (x, y), r, 1, -1)
        skel = result

    return (skel > 0).astype(np.uint8) * 255


# ── 最终平滑 ──────────────────────────────────────────────────

def smooth_edges(binary: np.ndarray) -> np.ndarray:
    """高斯模糊 + 重新二值化, 消除锯齿边缘."""
    if SMOOTH_SIGMA <= 0 or not binary.any():
        return binary
    f = binary.astype(np.float32)
    blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=SMOOTH_SIGMA,
                               borderType=cv2.BORDER_REPLICATE)
    _, result = cv2.threshold(blurred, SMOOTH_THRESHOLD, 255, cv2.THRESH_BINARY)
    return result.astype(np.uint8)


# ╔══════════════════════════════════════════════════════════════╗
# ║  四、完整检测管线                                          ║
# ╚══════════════════════════════════════════════════════════════╝

def weak_scratch_detection_pipeline(
    gray: np.ndarray,
    method: str = "all",
    verbose: bool = True,
) -> np.ndarray:
    """神经元纤维识别完整管线.

    根据 method 参数选择单一方法或融合所有方法（纤维识别）:
      - "bandpass":  仅频域带通滤波
      - "gabor":     仅 Gabor 方向滤波
      - "tophat":    仅 Top-Hat 形态学
      - "hessian":   仅 Hessian 脊线检测
      - "all":       融合以上四种方法 (推荐)

    Steps:
      1. 各方法独立计算响应图
      2. 加权融合响应图
      3. 二值化
      4. 形态学清理
      5. 伸长率过滤 (保留纤维状组件)
      6. 趋势线降噪 (PCA 拟合主趋势线, 消除偏离太远的孤立噪声块)
      7. 骨架化 — 将离散点聚合成连续线 (线可拐弯、可分裂)
      8. 边缘平滑

    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (uint8).
    method : str
        检测方法: "bandpass" | "gabor" | "tophat" | "hessian" | "all".
    verbose : bool
        是否打印中间统计.

    Returns
    -------
    result : np.ndarray
        二值 mask (uint8, 0/255).
    """
    if verbose:
        print(f"  [input] shape={gray.shape}, range=[{gray.min()}, {gray.max()}]")

    responses: dict[str, np.ndarray] = {}

    # ── 按需计算各方法响应 ──
    if method in ("bandpass", "all"):
        if verbose:
            print(f"  [bandpass] low={BP_LOW_CUT}, high={BP_HIGH_CUT}, order={BP_ORDER}")
        responses["bandpass"] = bandpass_filter(gray)

    if method in ("gabor", "all"):
        if verbose:
            print(f"  [gabor] thetas={len(GABOR_THETAS)} directions, "
                  f"σ={GABOR_SIGMA}, λ={GABOR_LAMBDA}, γ={GABOR_GAMMA}")
        responses["gabor"] = gabor_filter_bank(gray)

    if method in ("tophat", "all"):
        if verbose:
            print(f"  [tophat] kernel={TOPHAT_KERNEL_W}x{TOPHAT_KERNEL_H}, "
                  f"method={TOPHAT_METHOD}, multi-angle")
        responses["tophat"] = multi_angle_tophat(gray)

    if method in ("hessian", "all"):
        if verbose:
            print(f"  [hessian] scales={HESSIAN_SCALES}, mode={HESSIAN_MODE}, β={LINE_BETA}")
        responses["hessian"] = multi_scale_hessian(gray)

    if method not in responses and method != "all":
        raise ValueError(f"Unknown method: {method}")

    # ── 融合 ──
    if len(responses) == 1:
        fused = next(iter(responses.values()))
    else:
        weights = {
            "bandpass": WEIGHT_BANDPASS,
            "gabor":    WEIGHT_GABOR,
            "tophat":   WEIGHT_TOPHAT,
            "hessian":  WEIGHT_HESSIAN,
        }
        fused = fuse_responses(responses, weights)
        if verbose:
            print(f"  [fuse] weights: { {k: v for k, v in weights.items() if k in responses} }")

    # ── 二值化 ──
    binary = binarize_response(fused)
    fg_pct = binary.sum() / (binary.size * 255) * 100
    if verbose:
        print(f"  [binarize] method={BINARIZE_METHOD}, foreground={fg_pct:.2f}%")

    if not binary.any():
        if verbose:
            print("  [warn] no foreground after binarization, returning empty")
        return binary

    # ── 形态学清理 ──
    cleaned = morph_cleanup(binary)
    if verbose:
        fg2 = cleaned.sum() / (cleaned.size * 255) * 100
        print(f"  [morph] close({CLOSE_KERNEL}x{CLOSE_ITER}) + "
              f"open({OPEN_KERNEL}x{OPEN_ITER}), foreground={fg2:.2f}%")

    if not cleaned.any():
        if verbose:
            print("  [warn] no foreground after morph cleanup")
        return binary

    # ── 伸长率过滤 ──
    filtered = filter_elongated(cleaned)
    if verbose:
        fg3 = filtered.sum() / max(filtered.size * 255, 1) * 100
        n_comp = label(filtered > 0, structure=np.ones((3, 3), dtype=bool))[1]
        print(f"  [elongation] min_area={MIN_COMPONENT_AREA}, "
              f"min_elong={ELONGATION_MIN}, kept={n_comp} components, "
              f"foreground={fg3:.2f}%")

    # ── 趋势线降噪: 消除偏离连线趋势太远的色块 ──
    denoised = denoise_outliers(filtered)
    if verbose:
        fg4 = denoised.sum() / max(denoised.size * 255, 1) * 100
        n_comp_after = label(denoised > 0, structure=np.ones((3, 3), dtype=bool))[1]
        print(f"  [denoise] enable={DENOISE_ENABLE}, "
              f"kept={n_comp_after} components, foreground={fg4:.2f}%")

    if not denoised.any():
        if verbose:
            print("  [warn] no foreground after denoise, falling back to filtered")
        denoised = filtered

    # ── 骨架化: 将离散点聚合成连续线 (线可拐弯、可分裂) ──
    gathered = gather_onto_lines(denoised)
    if verbose:
        fg5 = gathered.sum() / max(gathered.size * 255, 1) * 100
        n_skel = label(gathered > 0, structure=np.ones((3, 3), dtype=bool))[1]
        print(f"  [gather] enable={GATHER_ENABLE}, close_k={int(min(denoised.shape)*GATHER_CLOSE_RATIO)}, "
              f"kept={n_skel} skeletons, foreground={fg5:.2f}%")

    if not gathered.any() and denoised.any():
        if verbose:
            print("  [warn] skeleton empty, using denoised mask")
        gathered = denoised

    # ── 最终输出: 骨架已是最优形态, 跳过边缘平滑; 否则平滑原始降噪结果 ──
    if GATHER_ENABLE:
        result = gathered
        if verbose:
            print(f"  [result] skeleton mode (skip smooth), "
                  f"fg={gathered.sum()/max(gathered.size*255,1)*100:.2f}%")
    else:
        result = smooth_edges(denoised)

    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║  五、I/O 和批处理                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def process_image(
    input_path: Path,
    output_path: Path,
    method: str = "all",
    verbose: bool = True,
) -> str:
    """读取图像 → 检测管线 → 保存结果. 返回状态信息."""
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return f"  [error] cannot read {input_path.name}"

    result = weak_scratch_detection_pipeline(gray, method=method, verbose=verbose)

    cv2.imwrite(str(output_path), result)

    fg_px = int(result.sum() / 255)
    total_px = result.size
    return (
        f"  ✓ {input_path.name} → {output_path.name}"
        f"  ({gray.shape[0]}×{gray.shape[1]}, "
        f"fg={fg_px}/{total_px} = {fg_px / total_px * 100:.1f}%)"
    )


def _process_wrapper(args: tuple) -> str:
    input_path, output_path, method, verbose = args
    return process_image(input_path, output_path, method, verbose)


# ╔══════════════════════════════════════════════════════════════╗
# ║  六、主入口                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="显微镜灰度图 — 高噪声背景下神经元纤维识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 默认: 融合所有方法, 批处理
  python main.py --generate               # 生成高噪声弱纤维训练数据
  python main.py --generate --num 500     # 生成 500 张
  python main.py --method gabor           # 仅使用 Gabor 纤维检测
  python main.py --method bandpass        # 仅使用频域带通滤波
  python main.py --method tophat          # 仅使用 Top-Hat 形态学
  python main.py --method hessian         # 仅使用 Hessian 脊线检测
  python main.py --input ./images --output ./out  # 指定 I/O 目录
        """,
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="生成高噪声弱纤维数据集 (而非运行检测管线)",
    )
    parser.add_argument(
        "--num", type=int, default=100,
        help="生成样本数 (与 --generate 配合使用, 默认 100)",
    )
    parser.add_argument(
        "--gen-dir", type=str, default=GEN_OUTPUT_DIR,
        help=f"数据生成输出目录 (默认: {GEN_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["all", "bandpass", "gabor", "tophat", "hessian"],
        help="检测方法 (默认: all, 融合所有方法纤维识别)",
    )
    parser.add_argument(
        "--input", type=str, default=INPUT_DIR,
        help=f"输入图像目录 (默认: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_DIR,
        help=f"结果输出目录 (默认: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"并行 worker 数 (默认: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式 (减少输出)",
    )

    args = parser.parse_args()

    # ── 数据生成模式 ──
    if args.generate:
        generate_dataset(num_samples=args.num, output_dir=args.gen_dir)
        return

    # ── 检测模式 ──
    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"错误: 输入目录不存在 → {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in exts
    )
    if not image_files:
        print(f"错误: {input_dir} 中无图像文件")
        sys.exit(1)

    n = len(image_files)
    verbose = not args.quiet

    print(f"找到 {n} 张图像")
    print(f"检测方法: {args.method}")
    if args.method == "all":
        print(f"  融合权重: bandpass={WEIGHT_BANDPASS}, gabor={WEIGHT_GABOR}, "
              f"tophat={WEIGHT_TOPHAT}, hessian={WEIGHT_HESSIAN}")
    print(f"  二值化: {BINARIZE_METHOD}")
    print(f"  形态学: close({CLOSE_KERNEL}x{CLOSE_ITER}) + open({OPEN_KERNEL}x{OPEN_ITER})")
    print(f"  伸长率: min_area={MIN_COMPONENT_AREA}, min_elong={ELONGATION_MIN}")
    if DENOISE_ENABLE:
        print(f"  降噪: 自适应趋势路径过滤 (PCA 确定性拟合)")
    print()

    tasks = []
    for img_path in image_files:
        out_path = output_dir / f"3Seg_{img_path.stem}.png"
        tasks.append((img_path, out_path, args.method, verbose))

    n_workers = min(len(tasks), args.workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_wrapper, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                t = futures[future]
                print(f"  [error] {t[0].name}: {e}")

    print(f"\n全部完成 → {output_dir}")


if __name__ == "__main__":
    main()
