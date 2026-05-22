"""
生成 256x256 模拟干涉条纹图
- 条纹方向、密度、形状随机
- 包含直线形、圆形、双核型、波纹型等
- 添加高斯噪声(标准差50) + 散斑噪声
- 带噪声灰度图输出到 1Den 文件夹，干净二值图输出到 1Den_clean 文件夹
- 按数字命名 png
"""

import os
import random

import numpy as np
from PIL import Image

OUT_DIR = "1Den"
CLEAN_DIR = "1Den_clean"
EDGE_DIR = "1Den_edge"
IMG_SIZE = 256
NUM_IMAGES = 10000
GAUSS_STD = 50  # 高斯噪声标准差


def generate_fringe(fringe_type: str, density: float, angle: float,
                    stress_points: list = None):
    """生成干涉条纹图，可选应力点（局部相位畸变）

    stress_points: list of (sx, sy, strength, sigma)

    返回: (fringe, phase)
          fringe  — cos²(phase)，值域 [0,1]
          phase   — 解析相位场，用于计算精确边缘图
    """
    h, w = IMG_SIZE, IMG_SIZE
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    # 基础条纹相位
    rad = angle * np.pi / 180.0
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    if fringe_type == "linear":
        phase = density * (cos_a * X + sin_a * Y)

    elif fringe_type == "circular":
        cx, cy = 0.5 + random.uniform(-0.2, 0.2), 0.5 + random.uniform(-0.2, 0.2)
        phase = density * np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    elif fringe_type == "bipolar":
        cx1, cy1 = 0.35, 0.5
        cx2, cy2 = 0.65, 0.5
        d1 = np.sqrt((X - cx1) ** 2 + (Y - cy1) ** 2)
        d2 = np.sqrt((X - cx2) ** 2 + (Y - cy2) ** 2)
        phase = density * (d1 - d2)

    elif fringe_type == "ripple":
        phase = density * (cos_a * X + sin_a * Y)
        # 叠加波纹调制
        freq = random.uniform(3.0, 8.0)
        phase += 2.0 * np.sin(freq * np.pi * (cos_a * X + sin_a * Y))

    elif fringe_type == "spiral":
        cx, cy = 0.5, 0.5
        phase = density * np.arctan2(Y - cy, X - cx)

    elif fringe_type == "mixed":
        # 随机混合两种类型
        sub_type = random.choice(["linear", "circular", "bipolar"])
        fringe, phase = generate_fringe(sub_type, density, angle, stress_points=None)
        return fringe, phase

    else:
        phase = density * (cos_a * X + sin_a * Y)

    # --- 添加应力点（局部高斯型相位畸变） ---
    if stress_points:
        stress_phase = np.zeros_like(X)
        for sx, sy, strength, sigma in stress_points:
            sx_norm = sx / IMG_SIZE
            sy_norm = sy / IMG_SIZE
            sigma_norm = sigma / IMG_SIZE
            dist2 = (X - sx_norm) ** 2 + (Y - sy_norm) ** 2
            stress_phase += strength * np.exp(-dist2 / (2 * sigma_norm ** 2))
        phase += stress_phase

    # 转条纹：cos^2(phase)，值域 [0, 1]
    fringe = np.cos(phase) ** 2
    return fringe, phase


def add_speckle(image: np.ndarray, strength: float = 0.3):
    """添加散斑噪声 (乘性噪声)"""
    speckle = np.random.exponential(scale=1.0, size=image.shape)
    # 归一化使均值保持为1
    speckle = speckle / np.mean(speckle)
    noisy = image * (1 + strength * (speckle - 1))
    return np.clip(noisy, 0, 1)


def generate_one():
    """生成一张完整的带噪声条纹图，随机添加0-4个应力点

    返回: (img_clean_binary, img_clean_cont, img_noisy, img_edge, fringe_type, density, angle, n_stress)
          img_clean_binary — 无噪声的二值化条纹图（0/255，用于 CLEAN_DIR）
          img_clean_cont   — 无噪声的连续 cos² 值（0-255 灰度，用于 U-Net 训练）
          img_noisy        — 添加噪声后的灰度图（用于 OUT_DIR）
          img_edge         — 解析边缘图（0-255，用于 EDGE_DIR）
    """
    fringe_types = ["linear", "circular", "bipolar", "ripple", "spiral", "mixed"]
    fringe_type = random.choice(fringe_types)

    density = random.uniform(2.0, 12.0)  # 条纹密度
    angle = random.uniform(0, 180)  # 方向角度

    # --- 随机生成0-4个应力点 ---
    r = random.random()
    if r < 0.20:
        num_stress = 0
    elif r < 0.55:
        num_stress = 1
    elif r < 0.85:
        num_stress = 2
    elif r < 0.95:
        num_stress = 3
    else:
        num_stress = 4

    stress_points = []
    for _ in range(num_stress):
        sx = random.uniform(IMG_SIZE * 0.15, IMG_SIZE * 0.85)
        sy = random.uniform(IMG_SIZE * 0.15, IMG_SIZE * 0.85)
        strength = random.uniform(2.0, 8.0)
        sigma = random.uniform(25.0, 80.0)
        stress_points.append((sx, sy, strength, sigma))

    # 生成干净条纹 + 解析相位（含应力点，无噪声）
    fringe_clean, phase = generate_fringe(fringe_type, density, angle, stress_points)

    # --- 干净图二值化：cos² 条纹以 0.5 为阈值直接二值化 ---
    img_clean_binary = ((fringe_clean >= 0.5) * 255).astype(np.uint8)

    # --- 解析边缘图：|sin(2*phase)| → 0-255 ---
    edge_map = np.abs(np.sin(2 * phase))
    img_edge = (edge_map * 255).astype(np.uint8)

    # --- 在副本上添加噪声 ---
    fringe_noisy = fringe_clean.copy()

    # 添加散斑噪声
    speckle_strength = random.uniform(0.1, 0.5)
    fringe_noisy = add_speckle(fringe_noisy, speckle_strength)

    # 添加高斯噪声
    noise = np.random.normal(0, GAUSS_STD / 255.0, fringe_noisy.shape)
    fringe_noisy = np.clip(fringe_noisy + noise, 0, 1)

    # 转为 0-255 uint8
    img_noisy = (fringe_noisy * 255).astype(np.uint8)

    # ── 连续干净图（cos² 值，供 U-Net 训练用）──
    img_clean_cont = (fringe_clean * 255).astype(np.uint8)

    return img_clean_binary, img_clean_cont, img_noisy, img_edge, fringe_type, density, angle, len(stress_points)

OUT_DIR_CONT = "1Den_clean_cont"  # 连续值干净图（cos² 0-255 灰度）


def main():
    import shutil
    # 清理旧数据，避免文件累积和可能的损坏
    for d in [OUT_DIR, CLEAN_DIR, EDGE_DIR, OUT_DIR_CONT]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(EDGE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_CONT, exist_ok=True)

    print(f"生成 {NUM_IMAGES} 张干涉条纹图...")
    print(f"  带噪声灰度图 → {OUT_DIR}/")
    print(f"  干净二值图   → {CLEAN_DIR}/")
    print(f"  连续干净图   → {OUT_DIR_CONT}/")
    print(f"  解析边缘图   → {EDGE_DIR}/")
    for i in range(NUM_IMAGES):
        img_clean, img_clean_cont, img_noisy, img_edge, ftype, density, angle, n_stress = generate_one()
        # 带噪声灰度图保存到 1Den
        Image.fromarray(img_noisy, mode="L").save(os.path.join(OUT_DIR, f"{i}.png"))
        # 干净二值图保存到 1Den_clean
        Image.fromarray(img_clean, mode="L").save(os.path.join(CLEAN_DIR, f"{i}.png"))
        # 连续值干净图保存到 1Den_clean_cont（cos² 值 0-255 灰度，供 U-Net 训练）
        Image.fromarray(img_clean_cont, mode="L").save(os.path.join(OUT_DIR_CONT, f"{i}.png"))
        # 解析边缘图保存到 1Den_edge
        Image.fromarray(img_edge, mode="L").save(os.path.join(EDGE_DIR, f"{i}.png"))

        if (i + 1) % 10 == 0:
            print(f"  已生成 {i + 1}/{NUM_IMAGES}")

    print("完成！")


if __name__ == "__main__":
    main()
