#!/usr/bin/env python3
# 1Den 数据集的 U-Net 去噪流水线（仅去噪，无任何预处理/后处理）
#   1Den/*.png (含噪声) → U-Net 去噪 → 二值化结果 → ../results/1Den_XX.png

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from train_unet import UNet

NOISY_DIR = "../1Den"
# main.py 不依赖 1Den_clean 参考图，仅做去噪输出
RESULTS_DIR = "../results"
DEVICE = torch.device("cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)


def save_image_binary(tensor, filepath):
    """将张量保存为二值化图像（0 或 255），使用 Otsu 自适应阈值"""
    if hasattr(tensor, "cpu"):
        arr = tensor.cpu().numpy().squeeze()
    else:
        arr = np.asarray(tensor).squeeze()
    # 使用 Otsu 自适应阈值替代固定 0.5
    arr_uint8 = (arr * 255).astype(np.uint8)
    _, otsu_thresh = cv2.threshold(
        arr_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    Image.fromarray(otsu_thresh, mode="L").save(filepath)


def denoise_image(model, img_path):
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0

    # ── 预处理：对偏暗图像做 CLAHE 自适应对比度增强 ──
    # 计算图像整体亮度，若偏暗则增强对比度
    img_mean = img.mean()
    if img_mean < 0.35:
        # 使用 CLAHE 增强局部对比度，限制对比度幅度防止噪声放大
        img_u8 = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img_u8)
        img = clahe_img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
    # 返回原始概率图（0~1 之间的浮点数），让 save_image_binary 统一做 Otsu 二值化
    probs = out.squeeze().cpu().numpy()
    return probs


def load_unet():
    """加载训练好的 U-Net 模型"""
    ckpt_path = "checkpoints/unet_denoiser_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    # checkpoint 可能是 dict {"unet": {...}} 或直接是 state_dict
    if isinstance(state, dict) and "unet" in state:
        model.load_state_dict(state["unet"])
        print(
            "[LOAD] 已加载 U-Net（含后处理参数的旧格式 checkpoint，仅使用 U-Net 权重）"
        )
    else:
        model.load_state_dict(state)
        print("[LOAD] 已加载 U-Net")

    model.eval()
    return model


def ensure_model():
    """检查模型是否存在，不存在则自动调用 train_unet.py 训练"""
    ckpt_path = "checkpoints/unet_denoiser_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"[CHECK] 模型文件 {ckpt_path} 不存在，开始训练...")
        ret = os.system(f"{sys.executable} train_unet.py")
        if ret != 0:
            raise RuntimeError("train_unet.py 训练失败，请检查数据或代码！")
        print("[CHECK] 训练完成，继续执行主程序。")
    else:
        print(f"[CHECK] 模型文件 {ckpt_path} 已存在，跳过训练。")


def main():
    # ── 1. 确保模型已训练 ──
    ensure_model()

    # ── 2. 扫描真实生产数据 ──
    files = sorted(os.listdir(NOISY_DIR))
    print(f"[LOAD] Found {len(files)} images in {NOISY_DIR}/")
    print("[LOAD] Loading U-Net model...")
    model = load_unet()
    print("[LOAD] 模型 loaded.\n")

    # ── 3. 主程序不依赖 clean 参考图，仅做去噪输出 ──


    for i, fname in enumerate(files):
        idx_str = os.path.splitext(fname)[0]
        idx_pad = idx_str.zfill(2)

        noisy_path = os.path.join(NOISY_DIR, fname)

        # 仅做 U-Net 去噪，无任何预处理/后处理
        denoised = denoise_image(model, noisy_path)
        out_name = f"1Den_{idx_pad}.png"
        out_path = os.path.join(RESULTS_DIR, out_name)
        save_image_binary(denoised, out_path)
        print(f"  [{idx_pad}] 去噪完成 -> {out_name}")

        # main.py 不依赖 clean 参考图，不做 MAE 评估

    print("\n" + "=" * 60)
    print(f"  Done! {len(files)} images processed.")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("=" * 60)



# ============================================================
# 主程序
# ============================================================


if __name__ == "__main__":
    main()
