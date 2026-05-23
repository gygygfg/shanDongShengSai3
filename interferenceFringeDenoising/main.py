#!/usr/bin/env python3
# 1Den 数据集去噪流水线
#   1Den/*.png (含噪声) -> CLAHE -> 自适应放大 -> U-Net -> sigmoid -> Guided Filter
#   -> 放大分辨率Otsu二值化 -> INTER_NEAREST缩回 -> morph_median -> ../results/1Den_XX.png
#
#   核心逻辑：
#     train_unet.py（版本3）训练 U-Net，用边界加权MSE做去噪。
#     main.py 加载训练好的模型推理：
#       - CLAHE 对比度增强：暗图模型输出方差过低，Otsu 误判（黑%高达88%→恢复至49%）
#       - 自适应放大：细密条纹放大后U-Net更好分辨边界
#       - 放大分辨率上Otsu二值化 + INTER_NEAREST缩回：避免INTER_AREA平滑概率图导致条纹黏连
#       - Guided Filter 去毛刺：沿边缘方向线性回归，保留条纹锐度
#       - morph_median 边缘平滑：在原分辨率上处理，避免放大后的kernel过大

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
    """自适应多尺度去噪：读图 -> CLAHE -> 密度检测 -> 必要时放大 -> U-Net -> sigmoid -> Guided Filter -> 返回二值图

    CLAHE 预处理对暗图像至关重要：暗图（mean≈0.22）模型输出方差过低，Otsu 无法正确二值化。
    CLAHE 增强局部对比度后，模型输出方差足够支撑 Otsu 找到正确阈值（验证：暗图黑%从88%恢复到49%）。
    Guided Filter 沿边缘方向线性回归，保留条纹锐度的同时抹掉毛刺噪声。

    自适应放大策略：
      - 放大后在高分辨率上推理，U-Net 能更好分辨细密条纹边界
      - 放大分辨率上直接做二值化确定边界，再用 INTER_NEAREST 缩回
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

    # ---- 在放大分辨率上做二值化，再用 INTER_NEAREST 缩回 ----
    # 不在原图上用 INTER_AREA 缩概率图，因为概率平滑会导致细密条纹边界模糊后黏连
    img_u8 = (np.clip(probs, 0.0, 1.0) * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 缩回原尺寸（如果放大过）
    if scale_level > 1:
        orig_h = int(binary.shape[0] / (2 ** (scale_level - 1)))
        orig_w = int(binary.shape[1] / (2 ** (scale_level - 1)))
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 边缘平滑（在原分辨率上做；纹理越密平滑越轻，避免细密条纹被糊成色块）
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


def save_image_binary_adaptive(tensor, filepath):
    """将张量保存为二值图像，使用 Otsu 自适应阈值（应对模型输出值域偏移）"""
    if hasattr(tensor, "cpu"):
        arr = tensor.cpu().numpy().squeeze()
    else:
        arr = np.asarray(tensor).squeeze()

    arr_clipped = np.clip(arr, 0.0, 1.0)
    img_u8 = (arr_clipped * 255).astype(np.uint8)

    # Otsu 自适应阈值：自动计算最佳分割点，不依赖固定 0.5
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 二值边缘平滑
    binary = smooth_binary_edges(binary, method="morph_median")

    Image.fromarray(binary, mode="L").save(filepath)



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
        # 直接保存二值图（denoise_image 已在放大分辨率上完成 Otsu + morph_median）
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

