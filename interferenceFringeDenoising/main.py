#!/usr/bin/env python3
# 1Den 数据集去噪流水线
#   1Den/*.png (含噪声) -> CLAHE 对比度增强 -> U-Net -> sigmoid -> 二值化 -> 边缘平滑 -> ../results/1Den_XX.png
#
#   核心逻辑：
#     train_unet.py（版本3）训练 U-Net，用 L1+边缘损失做去噪。
#     main.py 加载训练好的模型，先 CLAHE 增强对比度再推理，最后二值化 + 边缘平滑输出。

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
MAX_SCALE_ITER = 4  # 最大放大次数（2^4 = 16x），防止无限循环

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---- 纹理密度 ----
def compute_texture_density(img: np.ndarray) -> float:
    """计算纹理密度：颜色梯度波动的最大值（与训练时一致）"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad_mag.max())


def load_density_threshold() -> float:
    """从训练产物中加载安全纹理密度阈值"""
    threshold_path = "checkpoints/density_threshold.json"
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        t = data["density_threshold"]
        print(f"[LOAD] 安全纹理密度阈值: {t:.4f}（来自 {threshold_path}）")
        return t
    else:
        print(f"[WARN] 未找到 {threshold_path}，使用默认阈值 0.0（不触发放大）")
        return 0.0


def smooth_binary_edges(binary_img: np.ndarray, method: str = "morph_median") -> np.ndarray:
    """对二值图做边缘平滑处理，输入输出都是 0/255 uint8"""
    if method == "morph":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return result

    elif method == "median":
        return cv2.medianBlur(binary_img, 5)

    elif method == "morph_median":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        result = cv2.medianBlur(result, 5)
        return result

    elif method == "gaussian":
        blur = cv2.GaussianBlur(binary_img.astype(np.float32), (5, 5), 1.0)
        return (blur > 127).astype(np.uint8) * 255

    else:
        return binary_img


def save_image_binary(tensor, filepath, smooth_method: str = "morph_median"):
    """将张量保存为二值化图像（0 或 255），固定阈值 0.5 + 边缘平滑"""
    if hasattr(tensor, "cpu"):
        arr = tensor.cpu().numpy().squeeze()
    else:
        arr = np.asarray(tensor).squeeze()

    arr_clipped = np.clip(arr, 0.0, 1.0)
    binary = (arr_clipped >= 0.5).astype(np.uint8) * 255

    # 边缘平滑
    binary_smooth = smooth_binary_edges(binary, method=smooth_method)

    Image.fromarray(binary_smooth, mode="L").save(filepath)


def denoise_image(model, img_path, density_threshold: float):
    """自适应多尺度去噪：读图 -> 始终 CLAHE 增强对比度 -> 密度检测 -> 必要时放大 -> U-Net -> sigmoid -> 返回概率图"""
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0

    # ---- 预处理：始终 CLAHE 对比度增强 ----
    img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_u8)
    img = clahe_img.astype(np.float32) / 255.0

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

    # 如果放大过，缩回原尺寸
    if scale_level > 1:
        orig_h = int(probs.shape[0] / (2 ** (scale_level - 1)))
        orig_w = int(probs.shape[1] / (2 ** (scale_level - 1)))
        probs = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return np.clip(probs, 0.0, 1.0)


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

    # ---- 4. 选择边缘平滑方法 ----
    SMOOTH_METHOD = "morph_median"

    scaled_count = 0
    for i, fname in enumerate(files):
        idx_str = os.path.splitext(fname)[0]
        idx_pad = idx_str.zfill(2)
        noisy_path = os.path.join(NOISY_DIR, fname)

        # 自适应多尺度去噪
        denoised = denoise_image(model, noisy_path, density_threshold)

        # 检测是否触发了放大
        img_raw = np.array(Image.open(noisy_path), dtype=np.float32) / 255.0
        raw_density = compute_texture_density(img_raw)
        scale_tag = ""
        if raw_density > density_threshold:
            scaled_count += 1
            scale_tag = f" [密度={raw_density:.3f}>{density_threshold:.3f}, 已放大处理]"

        # 二值化 + 边缘平滑后保存
        out_name = f"1Den_{idx_pad}.png"
        out_path = os.path.join(RESULTS_DIR, out_name)
        save_image_binary(denoised, out_path, smooth_method=SMOOTH_METHOD)
        print(f"  [{idx_pad}] 完成 -> {out_name}{scale_tag}")

    print("\n" + "=" * 60)
    print(f"  Done! {len(files)} images processed.")
    print(f"  其中 {scaled_count} 张因纹理过密触发了自适应放大。")
    print(f"  安全纹理密度阈值: {density_threshold:.4f}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"  Edge smoothing method: {SMOOTH_METHOD}")
    print("=" * 60)


# ============================================================
# 主程序
# ============================================================


if __name__ == "__main__":
    main()
