"""
生成 256x256 模拟干涉条纹图（均匀分布版）
- 条纹类型、密度、方向、应力点数量、噪声强度全部均匀分层采样
- 每种参数组合覆盖全面，避免训练数据偏向某类场景
- 带噪声灰度图输出到 1Den，干净连续图输出到 1Den_clean_cont
"""

import math
import os
import random
import shutil

import numpy as np
from PIL import Image

OUT_DIR = "1Den"
CLEAN_DIR = "1Den_clean"
EDGE_DIR = "1Den_edge"
OUT_DIR_CONT = "1Den_clean_cont"
IMG_SIZE = 256
NUM_IMAGES = 2000                # 总生成量
GAUSS_STD = 50

# ── 均匀分档参数 ──
FRINGE_TYPES = ["linear", "circular", "bipolar", "ripple", "spiral", "mixed"]  # 6 种
DENSITY_BINS = [(2,5), (5,9), (9,14), (14,20), (20,28)]          # 5 档，整体更密集
STRESS_COUNTS = [0, 1, 2, 3, 4]                                 # 5 种，各 20%
SPECKLE_BINS = [(0.10,0.20), (0.20,0.30), (0.30,0.40), (0.40,0.50)]  # 4 区间
ANGLE_BINS   = [(0,45), (45,90), (90,135), (135,180)]           # 4 区间


def generate_fringe(fringe_type, density, angle, stress_points=None):
    """生成干涉条纹图，可选应力点（局部相位畸变）

    返回: (fringe, phase) — fringe = cos²(phase) ∈ [0,1]
    """
    h, w = IMG_SIZE, IMG_SIZE
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

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
        freq = random.uniform(3.0, 8.0)
        phase += 2.0 * np.sin(freq * np.pi * (cos_a * X + sin_a * Y))

    elif fringe_type == "spiral":
        cx, cy = 0.5, 0.5
        phase = density * np.arctan2(Y - cy, X - cx)

    elif fringe_type == "mixed":
        sub_type = random.choice(["linear", "circular", "bipolar"])
        fringe, phase = generate_fringe(sub_type, density, angle, stress_points=None)
        return fringe, phase

    else:
        phase = density * (cos_a * X + sin_a * Y)

    # 添加应力点
    if stress_points:
        stress_phase = np.zeros_like(X)
        for sx, sy, strength, sigma in stress_points:
            sx_norm = sx / IMG_SIZE
            sy_norm = sy / IMG_SIZE
            sigma_norm = sigma / IMG_SIZE
            dist2 = (X - sx_norm) ** 2 + (Y - sy_norm) ** 2
            stress_phase += strength * np.exp(-dist2 / (2 * sigma_norm ** 2))
        phase += stress_phase

    fringe = np.cos(phase) ** 2
    return fringe, phase


def add_speckle(image, strength=0.3):
    """添加散斑噪声（乘性噪声）"""
    speckle = np.random.exponential(scale=1.0, size=image.shape)
    speckle = speckle / np.mean(speckle)
    noisy = image * (1 + strength * (speckle - 1))
    return np.clip(noisy, 0, 1)


def generate_one(ftype, density, angle, num_stress, speckle_strength):
    """按指定参数生成一张条纹图（不再内部随机化）

    返回: (img_clean_binary, img_clean_cont, img_noisy, img_edge, ftype, density, angle, n_stress)
    """
    # 生成应力点
    stress_points = []
    for _ in range(num_stress):
        sx = random.uniform(IMG_SIZE * 0.15, IMG_SIZE * 0.85)
        sy = random.uniform(IMG_SIZE * 0.15, IMG_SIZE * 0.85)
        strength = random.uniform(2.0, 8.0)
        sigma = random.uniform(25.0, 80.0)
        stress_points.append((sx, sy, strength, sigma))

    fringe_clean, phase = generate_fringe(ftype, density, angle, stress_points)

    # 干净二值图
    img_clean_binary = ((fringe_clean >= 0.5) * 255).astype(np.uint8)

    # 解析边缘图
    edge_map = np.abs(np.sin(2 * phase))
    img_edge = (edge_map * 255).astype(np.uint8)

    # 添加噪声
    fringe_noisy = fringe_clean.copy()
    fringe_noisy = add_speckle(fringe_noisy, speckle_strength)
    noise = np.random.normal(0, GAUSS_STD / 255.0, fringe_noisy.shape)
    fringe_noisy = np.clip(fringe_noisy + noise, 0, 1)

    img_noisy = (fringe_noisy * 255).astype(np.uint8)
    img_clean_cont = (fringe_clean * 255).astype(np.uint8)

    return (img_clean_binary, img_clean_cont, img_noisy, img_edge,
            ftype, density, angle, num_stress)


def _uniform_sample_bins(bins):
    """从一组 (lo, hi) 区间中等概率选一个，再在区间内均匀随机"""
    lo, hi = random.choice(bins)
    return random.uniform(lo, hi)


def main():
    # 清理旧数据
    for d in [OUT_DIR, CLEAN_DIR, EDGE_DIR, OUT_DIR_CONT]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(EDGE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_CONT, exist_ok=True)

    print(f"生成 {NUM_IMAGES} 张干涉条纹图（均匀分布）...")
    print(f"  带噪声灰度图 → {OUT_DIR}/")
    print(f"  干净二值图   → {CLEAN_DIR}/")
    print(f"  连续干净图   → {OUT_DIR_CONT}/")
    print(f"  解析边缘图   → {EDGE_DIR}/")

    # ── 均匀分层采样：每条维度等概率，确保所有场景覆盖均匀 ──
    for i in range(NUM_IMAGES):
        ftype = random.choice(FRINGE_TYPES)              # 6 种各 ~16.7%
        density = _uniform_sample_bins(DENSITY_BINS)     # 5 档各 ~20%
        angle = _uniform_sample_bins(ANGLE_BINS)         # 4 档各 ~25%
        num_stress = random.choice(STRESS_COUNTS)        # 0-4 各 ~20%
        speckle_strength = _uniform_sample_bins(SPECKLE_BINS)  # 4 档各 ~25%

        result = generate_one(ftype, density, angle, num_stress, speckle_strength)
        img_clean, img_clean_cont, img_noisy, img_edge = result[:4]

        Image.fromarray(img_noisy, mode="L").save(os.path.join(OUT_DIR, f"{i}.png"))
        Image.fromarray(img_clean, mode="L").save(os.path.join(CLEAN_DIR, f"{i}.png"))
        Image.fromarray(img_clean_cont, mode="L").save(os.path.join(OUT_DIR_CONT, f"{i}.png"))
        Image.fromarray(img_edge, mode="L").save(os.path.join(EDGE_DIR, f"{i}.png"))

        if (i + 1) % 200 == 0:
            print(f"  已生成 {i + 1}/{NUM_IMAGES}")

    # ── 统计分布 ──
    print(f"\n完成！共 {NUM_IMAGES} 张。参数分布:"
          f"\n  条纹类型:  {len(FRINGE_TYPES)} 种 × ~{NUM_IMAGES//len(FRINGE_TYPES)} 张"
          f"\n  密度区间:  {len(DENSITY_BINS)} 档 × ~{NUM_IMAGES//len(DENSITY_BINS)} 张"
          f"\n  应力点数:  {len(STRESS_COUNTS)} 种 × ~{NUM_IMAGES//len(STRESS_COUNTS)} 张"
          f"\n  散斑区间:  {len(SPECKLE_BINS)} 档 × ~{NUM_IMAGES//len(SPECKLE_BINS)} 张"
          f"\n  角度区间:  {len(ANGLE_BINS)} 档 × ~{NUM_IMAGES//len(ANGLE_BINS)} 张")


if __name__ == "__main__":
    main()
