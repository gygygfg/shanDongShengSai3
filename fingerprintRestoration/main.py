"""
指纹增强处理脚本
读取 ../2Rec 下的 11.png~20.png，使用 pyfing 和 fingerprint_enhancer 处理，
结果保存到 ../results 下，命名为 2Rec_11.png ~ 2Rec_20.png
"""

import os

# ---- 强制使用 CPU（禁用 CUDA GPU）----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 禁用 TensorFlow 的 GPU 内存分配和 XLA 编译
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2

# ---- fingerprint_enhancer (Oriented Gabor Filter Bank) ----
import fingerprint_enhancer
import numpy as np

# ---- pyfing (GBFEN - Gabor-Based Fingerprint Enhancement) ----
import pyfing as pf

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "2Rec")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_with_pyfing(img: np.ndarray) -> np.ndarray:
    """使用 pyfing 的 GBFEN 方法增强指纹"""
    # 使用传统方法（GMFS/GBFOE/XSFFE/GBFEN），避免加载 Keras 模型
    mask = pf.fingerprint_segmentation(img, method="GMFS")
    orientations = pf.orientation_field_estimation(img, mask, method="GBFOE")
    frequencies = pf.frequency_estimation(img, orientations, mask, method="XSFFE")
    enhanced = pf.fingerprint_enhancement(
        img, orientations, frequencies, mask, method="GBFEN"
    )
    return enhanced


def process_with_fingerprint_enhancer(img: np.ndarray) -> np.ndarray:
    """使用 fingerprint_enhancer 增强指纹"""
    return fingerprint_enhancer.enhance_fingerprint(img)


def main():
    for i in range(11, 21):  # 11 ~ 20
        input_path = os.path.join(INPUT_DIR, f"{i}.png")
        if not os.path.exists(input_path):
            print(f"[跳过] {input_path} 不存在")
            continue

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[错误] 无法读取 {input_path}")
            continue
        print(f"[处理] {i}.png ({img.shape[1]}x{img.shape[0]})")

        # 使用 pyfing GBFEN 方法增强
        enhanced = process_with_pyfing(img)

        # 二值化处理：将增强后的灰度图转为二值图
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        output_path = os.path.join(OUTPUT_DIR, f"2Rec_{i}.png")
        cv2.imwrite(output_path, binary)
        print(f"  -> 保存: {output_path}")

    print("\n全部处理完成！")


if __name__ == "__main__":
    main()
