#!/usr/bin/env python3
# 1Den 数据集去噪流水线
#   1Den/*.png (含噪声) → U-Net 去噪 → 二值化 → 边缘平滑 → ../results/1Den_XX.png

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from train_unet import UNet

NOISY_DIR = "../1Den"
RESULTS_DIR = "../results"
DEVICE = torch.device("cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)


def smooth_binary_edges(binary_img: np.ndarray, method: str = "morph_median") -> np.ndarray:
    """对二值图做边缘平滑处理，输入输出都是 0/255 uint8
    
    支持的方法:
      - "morph":     形态学闭运算→开运算（3x3椭圆核）
      - "median":    中值滤波 5x5
      - "morph_median": 形态学 + 中值滤波组合（推荐）
      - "gaussian":  高斯模糊→重阈值
    """
    if method == "morph":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return result

    elif method == "median":
        return cv2.medianBlur(binary_img, 5)

    elif method == "morph_median":
        # 先形态学去孤立噪点+填小孔，再中值滤波平滑边缘
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
    """将张量保存为二值化图像（0 或 255），固定阈值 0.5 + 边缘平滑

    改用固定阈值 0.5 替代 Otsu 自适应阈值，原因：
    - cos² 条纹天然以 0.5 为分界线（亮暗各 50% 占空比）
    - 密集条纹经 U-Net 去噪后直方图可能失去双峰性，Otsu 会选出错误阈值导致条纹缺失
    """
    if hasattr(tensor, "cpu"):
        arr = tensor.cpu().numpy().squeeze()
    else:
        arr = np.asarray(tensor).squeeze()

    # 使用固定阈值 0.5（归一化后），与生成时的阈值一致
    arr_clipped = np.clip(arr, 0.0, 1.0)
    binary = (arr_clipped >= 0.5).astype(np.uint8) * 255

    # 边缘平滑
    binary_smooth = smooth_binary_edges(binary, method=smooth_method)

    Image.fromarray(binary_smooth, mode="L").save(filepath)


def denoise_image(model, img_path, smooth_method: str = "morph_median"):
    """U-Net 去噪 → 二值化 → 边缘平滑，一步完成"""
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0

    # ── 预处理：对偏暗图像做 CLAHE 自适应对比度增强 ──
    img_mean = img.mean()
    if img_mean < 0.35:
        img_u8 = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img_u8)
        img = clahe_img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
    probs = out.squeeze().cpu().numpy()
    return probs  # 返回概率图，由 main() 统一调用 save_image_binary


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
    """检查模型是否已收敛，未收敛则自动调用 train_unet.py 训练"""
    final_path = "checkpoints/unet_denoiser_final.pth"
    best_path = "checkpoints/unet_denoiser_best.pth"

    # final.pth 存在 → 训练已完成（60 epoch 跑完），直接跳过
    if os.path.exists(final_path):
        print(f"[CHECK] 模型已收敛（{final_path} 存在），跳过训练。")
        return

    # best.pth 存在但 final.pth 不存在 → 训练中断但有可用模型，跳过训练
    if os.path.exists(best_path):
        print(f"[CHECK] 发现已有最佳模型 {best_path}，跳过训练直接使用。")
        return

    # 都不存在 → 需要训练
    print(f"[CHECK] 模型文件不存在，开始训练...")
    ret = os.system(f"{sys.executable} train_unet.py")
    if ret != 0:
        raise RuntimeError("train_unet.py 训练失败，请检查数据或代码！")
    print("[CHECK] 训练完成，继续执行主程序。")


def main():
    # ── 1. 确保模型已训练 ──
    ensure_model()

    # ── 2. 扫描真实生产数据 ──
    files = sorted(os.listdir(NOISY_DIR))
    print(f"[LOAD] Found {len(files)} images in {NOISY_DIR}/")
    print("[LOAD] Loading U-Net model...")
    model = load_unet()
    print("[LOAD] 模型 loaded.\n")

    # ── 3. 选择边缘平滑方法 ──
    # "morph_median": 形态学闭开运算 + 中值滤波（推荐，平滑效果好）
    # "morph":        仅形态学（保留更多细节）
    # "median":       仅中值滤波
    # "gaussian":     高斯模糊重阈值
    # "none":         不做平滑（原来的行为）
    SMOOTH_METHOD = "morph_median"

    for i, fname in enumerate(files):
        idx_str = os.path.splitext(fname)[0]
        idx_pad = idx_str.zfill(2)
        noisy_path = os.path.join(NOISY_DIR, fname)

        # U-Net 去噪，得到概率图
        denoised = denoise_image(model, noisy_path)

        # 二值化 + 边缘平滑后保存
        out_name = f"1Den_{idx_pad}.png"
        out_path = os.path.join(RESULTS_DIR, out_name)
        save_image_binary(denoised, out_path, smooth_method=SMOOTH_METHOD)
        print(f"  [{idx_pad}] 完成 -> {out_name} (平滑方法: {SMOOTH_METHOD})")

    print("\n" + "=" * 60)
    print(f"  Done! {len(files)} images processed.")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"  Edge smoothing method: {SMOOTH_METHOD}")
    print("=" * 60)



# ============================================================
# 主程序
# ============================================================


if __name__ == "__main__":
    main()
