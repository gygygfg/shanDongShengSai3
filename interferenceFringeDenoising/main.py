#!/usr/bin/env python3
# 1Den 数据集去噪流水线
#   1Den/*.png (含噪声) -> CLAHE -> 自适应放大 -> U-Net -> sigmoid -> Guided Filter
#   -> 放大分辨率Otsu二值化 -> 放大分辨率回归曲线边缘平滑 -> INTER_NEAREST缩回 -> morph_median
#   -> ../results/1Den_XX.png
#
#   核心逻辑：
#     train_unet.py（版本3）训练 U-Net，用边界加权MSE做去噪。
#     main.py 加载训练好的模型推理：
#       - CLAHE 对比度增强：暗图模型输出方差过低，Otsu 误判（黑%高达88%→恢复至49%）
#       - 自适应放大：细密条纹放大后U-Net更好分辨边界
#       - 放大分辨率上Otsu二值化 + 回归曲线边缘平滑 + INTER_NEAREST缩回
#       - Guided Filter 去毛刺：沿边缘方向线性回归，保留条纹锐度
#       - 回归曲线边缘平滑在放大分辨率上做，轮廓精度高、效果远好于缩回后修补锯齿

import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from train_unet import UNet

NOISY_DIR = "../1Den"
RESULTS_DIR = "../results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SCALE_ITER = 6  # 最大放大次数（2^6 = 64x），防止无限循环

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---- 纹理密度 ----
def compute_texture_density(img: np.ndarray) -> float:
    """计算纹理密度：颜色梯度波动的最大值（与训练时一致）"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad_mag.max())


def denoise_image(model, img_path, density_threshold: float):
    """自适应多尺度去噪：读图 -> CLAHE -> 密度检测 -> 必要时放大 -> U-Net -> sigmoid
    -> Guided Filter -> 放大分辨率Otsu二值化 -> 放大分辨率回归曲线边缘平滑 -> INTER_NEAREST缩回

    CLAHE 预处理对暗图像至关重要：暗图（mean≈0.22）模型输出方差过低，Otsu 无法正确二值化。
    CLAHE 增强局部对比度后，模型输出方差足够支撑 Otsu 找到正确阈值（验证：暗图黑%从88%恢复到49%）。
    Guided Filter 沿边缘方向线性回归，保留条纹锐度的同时抹掉毛刺噪声。

    自适应放大策略：
      - 放大后在高分辨率上推理，U-Net 能更好分辨细密条纹边界
      - 放大分辨率上直接做二值化 + 回归曲线边缘平滑，再用 INTER_NEAREST 缩回
      - 回归曲线边缘平滑在放大分辨率上做：轮廓点密集，SG平滑效果远好于缩回后修补锯齿
      - 避免 INTER_AREA 对概率图的平滑导致细密条纹黏连
    """
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0

    # ---- CLAHE 对比度增强（训练时不包含此步骤，但对暗图测试集至关重要）----
    img_u8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_u8 = clahe.apply(img_u8)
    img = img_u8.astype(np.float32) / 255.0

    scale_level = 1
    density = compute_texture_density(img)
    while density > density_threshold and scale_level <= MAX_SCALE_ITER:
        scale_level += 1
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                         interpolation=cv2.INTER_LINEAR)
        density = compute_texture_density(img)

    # 推理（模型输出 logits，sigmoid 转概率）
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # ---- Guided Filter 边缘回归去毛刺 ----
    # eps=0.001（而非 0.01）：暗图（如08）在 eps=0.01 下 Otsu 阈值漂移严重（黑% 52→94），
    # eps=0.001 几乎不改变输出分布，仍能有效去毛刺
    # 自适应 radius：纹理越密 → 窗口越小，避免把细密条纹糊成色块
    gf_radius_map = {1: 4, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1}
    gf_radius = gf_radius_map.get(scale_level, 1)
    probs = guided_filter(probs, radius=gf_radius, eps=0.001)

    # ---- 在放大分辨率上做二值化 ----
    # 不在原图上用 INTER_AREA 缩概率图，因为概率平滑会导致细密条纹边界模糊后黏连
    img_u8 = (np.clip(probs, 0.0, 1.0) * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- 回归曲线边缘平滑：在放大分辨率上做，轮廓细节丰富、效果远好于缩回后修补锯齿 ----
    # 窗口大小按放大倍数等比例缩放；高倍时配合轮廓重采样，51 足够覆盖物理范围
    enlargement = 2 ** (scale_level - 1)
    reg_window_map = {1: 21, 2: 31, 3: 41, 4: 41, 5: 51, 6: 51}
    reg_window = reg_window_map.get(scale_level, 21)
    binary = regression_edge_smooth(binary, window_size=reg_window)

    # 缩回原尺寸（平滑后再缩回，INTER_NEAREST 不会重新引入锯齿）
    if scale_level > 1:
        orig_h = int(binary.shape[0] / enlargement)
        orig_w = int(binary.shape[1] / enlargement)
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 边缘平滑（在原分辨率上做轻量清理）
    binary = smooth_binary_edges(binary, method="morph_median", scale_level=scale_level)

    return binary


def load_density_threshold() -> float:
    """从训练产物中加载安全纹理密度阈值（生产环境调整为原来的0.6倍）"""
    threshold_path = "checkpoints/density_threshold.json"
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        t = data["density_threshold"]
        t = t * 0.6  # 生产环境阈值调整为原来的0.6倍
        print(f"[LOAD] 安全纹理密度阈值: {t:.4f}（来自 {threshold_path}，调整为原值的0.6倍）")
        return t
    else:
        print(f"[WARN] 未找到 {threshold_path}，使用默认阈值 0.0（不触发放大）")
        return 0.0


# ---- Guided Filter 边缘回归去毛刺 ----
def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """盒式滤波（均值滤波）。ksize 必须为奇数，避免 anchor 偏移导致相位错位（马赛克根因）"""
    ksize = (2 * radius + 1, 2 * radius + 1)
    return cv2.blur(img, ksize)


def guided_filter(p: np.ndarray, radius: int = 4, eps: float = 0.01) -> np.ndarray:
    """自引导滤波：沿边缘方向做线性回归平滑，保留条纹锐度，抹掉毛刺噪声。

    每个局部窗口拟合线性模型 q_i = a_k * I_i + b_k：
      - 边缘区域（引导图方差大）→ a ≈ 1，保留锐度
      - 平坦区域（方差小）→ a ≈ 0，抹平噪声

    参数:
        radius: 窗口半径（窗口大小 = (2*radius+1)²），默认 4 → 9×9
        eps:    正则化系数，越大则平滑越强，默认 0.01
    """
    mean_p = box_filter(p, radius)
    mean_pp = box_filter(p * p, radius)
    var_p = mean_pp - mean_p * mean_p

    # 线性回归系数: a = var / (var + eps), b = mean - a * mean
    a = var_p / (var_p + eps)
    b = mean_p - a * mean_p

    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    q = mean_a * p + mean_b
    return np.clip(q, 0.0, 1.0)



def remove_small_components(binary_img: np.ndarray, min_size: int = 7) -> np.ndarray:
    """连通组件去噪：去除二值图中孤立的小连通区域（椒盐噪声），天然保留条纹边缘。

    分别处理白色小区域（暗区噪点）和黑色小区域（亮区噪点）。
    连通组件分析逐像素判断连通性，不会模糊边缘，适用于任意纹理密度。
    """
    result = binary_img.copy()

    # 去除白色孤立噪点（暗区中的白点）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            result[labels == i] = 0

    # 去除黑色孤立噪点（亮区中的黑点）
    inverted = cv2.bitwise_not(result)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            result[labels == i] = 255

    return result


def _savgol_smooth_1d(y: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
    """一维 Savitzky-Golay 滤波器：对序列 y 做滑动窗口多项式回归。

    本质是局部最小二乘多项式拟合，等价于在边缘轮廓上做「回归曲线平滑」。
    纯 numpy 实现，不依赖 scipy。

    使用周期性填充处理闭合轮廓：在两端各补 window_size 长度的镜像数据，
    使平滑结果在首尾衔接处不产生断裂。
    """
    n = len(y)
    if n < window_size:
        return y.copy()
    half = window_size // 2

    # 周期性填充：首尾各补 window_size 个点，保证闭合轮廓平滑后衔接自然
    pad = window_size
    y_padded = np.concatenate([y[-pad:], y, y[:pad]])
    n_padded = len(y_padded)

    result = np.empty(n_padded, dtype=y.dtype)
    for i in range(n_padded):
        left = max(0, i - half)
        right = min(n_padded, i + half + 1)
        x_local = np.arange(left - i, right - i, dtype=np.float64)
        y_local = y_padded[left:right]
        if len(x_local) <= poly_order:
            result[i] = y_padded[i]
        else:
            coeffs = np.polyfit(x_local, y_local, poly_order)
            result[i] = np.polyval(coeffs, 0.0)

    # 截取原始范围
    return result[pad:pad + n]


def _gaussian_smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    """一维高斯平滑（卷积），用于去除轮廓坐标的量化锯齿。

    sigma 控制平滑强度：越大越平滑。边界处不做卷积，保持原值。
    """
    n = len(y)
    if n < 3 or sigma < 0.5:
        return y.copy()

    # 核半径取 3*sigma（覆盖 99.7%），最小 1，最大不超过 n//2
    radius = max(1, min(int(3.0 * sigma), n // 2))
    ksize = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel /= kernel.sum()

    result = np.convolve(y, kernel, mode='same')

    # 边界不做卷积，保持原值（对闭合轮廓来说这些位置会被周期性 SG 平滑修正）
    result[:radius] = y[:radius]
    result[-radius:] = y[-radius:]
    return result


def _resample_contour_uniform(pts: np.ndarray, target_n: int) -> np.ndarray:
    """按弧长均匀重采样轮廓点，消除 CHAIN_APPROX_NONE 的冗余密集采样。

    长轮廓重采样后，同样 window_size 能覆盖更大的物理范围，
    对高倍放大图像的平滑效果提升显著。
    """
    n = len(pts)
    if n <= target_n:
        return pts

    diffs = np.diff(pts, axis=0)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.concatenate([[0.0], np.cumsum(dists)])
    total = cum_dist[-1]

    if total < 1e-6:
        return pts[::max(1, n // target_n)]

    sample_dists = np.linspace(0, total, target_n)
    new_pts = np.empty((target_n, 2), dtype=pts.dtype)
    for d in range(2):
        new_pts[:, d] = np.interp(sample_dists, cum_dist, pts[:, d])
    return new_pts


def regression_edge_smooth(binary_img: np.ndarray,
                            window_size: int = 11,
                            poly_order: int = 3) -> np.ndarray:
    """对二值图的每条轮廓做 Savitzky-Golay 回归曲线平滑，基于连通组件重建填充区域。

    算法（v3）：
      1. 提取所有轮廓，长轮廓按弧长均匀重采样
      2. 高斯预平滑去除量化锯齿 → 周期性 SG 回归平滑（两遍：粗→细）
      3. 将平滑后的轮廓画到空白画布上作为边界线（barrier）
      4. 对非边界区域做连通组件分析
      5. 每个连通区域从原始二值图中采样，按多数投票决定填充颜色
      6. 边界像素通过膨胀从相邻区域分配颜色

    天然支持环状曲线、螺旋等任意拓扑结构，不依赖 RETR_CCOMP 层级关系。

    参数:
        binary_img:  输入二值图 (0/255 uint8)
        window_size: SG 滑动窗口大小（奇数），越大平滑越强
        poly_order:  多项式阶数，3=cubic 能拟合 S 弯（默认 3）
    """
    if poly_order >= window_size:
        poly_order = window_size - 1
    if poly_order < 1:
        poly_order = 1
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, 101)  # 放宽上限，配合重采样

    contours, _hierarchy = cv2.findContours(binary_img.copy(),
                                             cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_NONE)
    if not contours:
        return binary_img.copy()

    h, w = binary_img.shape

    # ---- 步骤 1：画平滑边界 ----
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    for contour in contours:
        pts = contour[:, 0, :].astype(np.float64)
        n_pts = len(pts)

        # 长轮廓重采样：使 window_size 覆盖更大的物理范围
        if n_pts > 2000:
            pts = _resample_contour_uniform(pts, 1500)
            n_pts = len(pts)

        if n_pts < window_size:
            smoothed = contour  # 太短不处理，保持原样
        else:
            x, y = pts[:, 0], pts[:, 1]

            # 高斯预平滑：去除整数像素坐标的量化锯齿
            sigma = max(window_size / 6.0, 1.5)
            x = _gaussian_smooth_1d(x, sigma)
            y = _gaussian_smooth_1d(y, sigma)

            # 第一遍 SG（粗平滑）：大窗口捕捉整体曲线趋势
            x = _savgol_smooth_1d(x, window_size, poly_order)
            y = _savgol_smooth_1d(y, window_size, poly_order)

            # 第二遍 SG（细平滑）：小窗口修正局部波动
            ws2 = max(window_size // 3, 7)
            if ws2 % 2 == 0:
                ws2 += 1
            po2 = min(poly_order, ws2 - 1)
            x = _savgol_smooth_1d(x, ws2, po2)
            y = _savgol_smooth_1d(y, ws2, po2)

            smooth_pts = np.stack([x, y], axis=1).round().astype(np.int32)
            smoothed = smooth_pts.reshape(-1, 1, 2)

        cv2.polylines(boundary_mask, [smoothed], isClosed=True, color=1, thickness=2)

    # ---- 步骤 2：连通组件分析（非边界区域） ----
    non_boundary = (1 - boundary_mask).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(non_boundary, connectivity=8)

    result = np.zeros((h, w), dtype=np.uint8)

    # ---- 步骤 3：每个连通区域从原图多数投票决定颜色 ----
    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        region_vals = binary_img[region_mask]
        if region_vals.size == 0:
            continue
        fill_value = 255 if np.mean(region_vals) > 127 else 0
        result[region_mask] = fill_value

    # ---- 步骤 4：边界像素分配给相邻区域（膨胀覆盖） ----
    unfilled = (result == 0)
    if unfilled.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(result, kernel, iterations=1)
        result[unfilled] = dilated[unfilled]

        unfilled2 = (result == 0)
        if unfilled2.any():
            dilated2 = cv2.dilate(result, kernel, iterations=1)
            result[unfilled2] = dilated2[unfilled2]

    return result


def smooth_binary_edges(binary_img: np.ndarray, method: str = "morph_median",
                        scale_level: int = 1) -> np.ndarray:
    """对二值图做边缘平滑处理，输入输出都是 0/255 uint8。

    纹理密度越高（scale_level 越大），平滑越轻，避免细密条纹被糊成色块：
      - scale_level=1（纹理不密，未放大）：morph_close+open + medianBlur(5)，充分去毛刺
      - scale_level>=2（纹理偏密/非常密）：连通组件去噪替代 morph 操作
        连通组件天然保留边缘，既能清除稀疏区域的孤立噪点，又不会把细密条纹糊成色块
    """
    if method == "morph":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return result

    elif method == "median":
        return cv2.medianBlur(binary_img, 5)

    elif method == "morph_median":
        if scale_level <= 1:
            # 纹理不密：充分平滑去毛刺
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            result = cv2.medianBlur(result, 5)
        else:
            # 纹理偏密/非常密：连通组件去噪，边缘无损
            # scale_level 越高 → min_size 越小（纹理越密，只清除更小的孤立噪点）
            min_size = 9 if scale_level == 2 else 5
            result = remove_small_components(binary_img, min_size=min_size)
        return result

    elif method == "gaussian":
        blur = cv2.GaussianBlur(binary_img.astype(np.float32), (5, 5), 1.0)
        return (blur > 127).astype(np.uint8) * 255

    else:
        return binary_img






def load_unet():
    """加载训练好的 U-Net 模型"""
    ckpt_path = "checkpoints/unet_denoiser_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 版本3的 train_unet.py 直接保存 model.state_dict()
    model.load_state_dict(state)
    print("[LOAD] 已加载 U-Net（版本3 checkpoint）")

    model.eval()
    return model


def ensure_model():
    """检查模型是否已收敛，未收敛则自动调用 train_unet.py 训练"""
    final_path = "checkpoints/unet_denoiser_final.pth"
    best_path = "checkpoints/unet_denoiser_best.pth"

    if os.path.exists(final_path):
        print(f"[CHECK] 模型已收敛（{final_path} 存在），跳过训练。")
        return

    if os.path.exists(best_path):
        print(f"[CHECK] 发现已有最佳模型 {best_path}，跳过训练直接使用。")
        return

    print(f"[CHECK] 模型文件不存在，开始训练...")
    ret = os.system(f"{sys.executable} train_unet.py")
    if ret != 0:
        raise RuntimeError("train_unet.py 训练失败，请检查数据或代码！")
    print("[CHECK] 训练完成，继续执行主程序。")


def main():
    # ---- 1. 确保模型已训练 ----
    ensure_model()

    # ---- 2. 加载安全纹理密度阈值 ----
    density_threshold = load_density_threshold()

    # ---- 3. 扫描真实生产数据 ----
    files = sorted(os.listdir(NOISY_DIR))
    print(f"[LOAD] Found {len(files)} images in {NOISY_DIR}/")
    print("[LOAD] Loading U-Net model...")
    model = load_unet()
    print("[LOAD] 模型 loaded.\n")

    scaled_count = 0
    for i, fname in enumerate(files):
        idx_str = os.path.splitext(fname)[0]
        idx_pad = idx_str.zfill(2)
        noisy_path = os.path.join(NOISY_DIR, fname)

        # 自适应多尺度去噪（返回二值图）
        binary = denoise_image(model, noisy_path, density_threshold)

        # 检测是否触发了放大
        img_raw = np.array(Image.open(noisy_path), dtype=np.float32) / 255.0
        raw_density = compute_texture_density(img_raw)
        scale_tag = ""
        if raw_density > density_threshold:
            scaled_count += 1
            scale_tag = f" [密度={raw_density:.3f}>{density_threshold:.3f}, 已放大处理]"


        out_name = f"1Den_{idx_pad}.png"
        out_path = os.path.join(RESULTS_DIR, out_name)
        # 直接保存二值图（denoise_image 已在放大分辨率上完成 Otsu + regression_edge_smooth + morph_median）
        Image.fromarray(binary, mode="L").save(out_path)
        print(f"  [{idx_pad}] 完成 -> {out_name}{scale_tag}")

    print("\n" + "=" * 60)
    print(f"  Done! {len(files)} images processed.")
    print(f"  其中 {scaled_count} 张因纹理过密触发了自适应放大。")
    print(f"  安全纹理密度阈值: {density_threshold:.4f}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


# ============================================================
# 主程序
# ============================================================


if __name__ == "__main__":
    main()

