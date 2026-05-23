"""
在现有条纹数据上训练 U-Net 去噪模型（1Den -> 1Den_clean_cont）
"""

import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── U-Net 架构 ─────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.ups = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


# ── 数据集（支持自适应多尺度放大）────────────────────────────────────
# ── 自适应放大配置 ──
MAX_SCALE_ITER = 6  # 最大放大次数（2^6 = 64x），与 main.py 保持一致


class FringeDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, edge_dir=None,
                 density_threshold: float = 0.0,
                 safety_ratio: float = 1.0,
                 max_scale_iter: int = MAX_SCALE_ITER):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.edge_dir = edge_dir
        self.files = sorted(os.listdir(noisy_dir))
        self.density_threshold = density_threshold
        self.safety_ratio = safety_ratio
        self.max_scale_iter = max_scale_iter

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy = np.array(Image.open(os.path.join(self.noisy_dir, fname)), dtype=np.float32) / 255.0
        clean = np.array(Image.open(os.path.join(self.clean_dir, fname)), dtype=np.float32) / 255.0
        edge = None
        if self.edge_dir is not None:
            edge = np.array(Image.open(os.path.join(self.edge_dir, fname)), dtype=np.float32) / 255.0

        # ── 自适应多尺度放大：与 main.py denoise_image 逻辑一致 ──
        # 纹理密度超过阈值时逐级 2x 放大，直到密度回落或达到最大放大次数
        if self.density_threshold > 0:
            effective_threshold = self.density_threshold * self.safety_ratio
            scale_level = 1
            density = compute_texture_density(clean)
            while density > effective_threshold and scale_level <= self.max_scale_iter:
                scale_level += 1
                new_h, new_w = noisy.shape[0] * 2, noisy.shape[1] * 2
                noisy = cv2.resize(noisy, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                clean = cv2.resize(clean, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if edge is not None:
                    edge = cv2.resize(edge, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                density = compute_texture_density(clean)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        clean = torch.from_numpy(clean).unsqueeze(0)
        if edge is not None:
            edge = torch.from_numpy(edge).unsqueeze(0)
            return noisy, clean, edge
        return noisy, clean


# ── 变尺寸批处理 ─────────────────────────────────────────────
def collate_variable_size(batch):
    """将不同尺寸的图像 zero-padding 到 batch 内最大尺寸，支持自适应放大后的变尺寸批处理"""
    has_edge = len(batch[0]) == 3

    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)

    noisy_list, clean_list, edge_list = [], [], []
    for item in batch:
        n, c = item[0], item[1]
        pad_h = max_h - n.shape[1]
        pad_w = max_w - n.shape[2]
        if pad_h > 0 or pad_w > 0:
            n = F.pad(n, (0, pad_w, 0, pad_h))
            c = F.pad(c, (0, pad_w, 0, pad_h))
        noisy_list.append(n)
        clean_list.append(c)
        if has_edge:
            e = item[2]
            if pad_h > 0 or pad_w > 0:
                e = F.pad(e, (0, pad_w, 0, pad_h))
            edge_list.append(e)

    noisy = torch.stack(noisy_list)
    clean = torch.stack(clean_list)
    if has_edge:
        edge = torch.stack(edge_list)
        return noisy, clean, edge
    return noisy, clean


# ── 边界感知损失 ─────────────────────────────────────────────
def edge_weighted_mse_loss(pred, target, edge_map, alpha=3.0):
    """
    边界加权的 MSE 损失：在黑白分界处（edge_map 值高）给予更高权重，
    迫使模型精确还原条纹边界位置。

    edge_map = |sin(2*phase)| ∈ [0, 1]，边界处趋近 1。
    weight  = 1 + alpha * edge_map
    loss    = mean(weight * (pred - target)^2)
    """
    weight = 1.0 + alpha * edge_map
    diff = (pred - target) ** 2
    return (weight * diff).mean()


# ── 纹理密度计算 ──
def compute_texture_density(img: np.ndarray) -> float:
    """计算纹理密度：颜色梯度波动的最大值"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return float(grad_mag.max())


# ── 训练 ──
def train(train_count=100, val_count=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_ch=1, out_ch=1).to(device)
    mse_criterion = nn.MSELoss()  # 用于监控基础 MSE
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    edge_weight = 3.0  # 边界加权系数
    TRAIN_SAFETY_RATIO = 1.0  # 训练时使用 P95 原始阈值（生产环境用 0.8 更保守）

    # ════════════════════════════════════════════════════════════
    # 第一步：从全部 clean 图像计算全局密度分布，确定安全阈值
    # ════════════════════════════════════════════════════════════
    print("正在计算全局密度分布...")
    all_clean_files = sorted(os.listdir("1Den_clean_cont"))
    densities = []
    for fname in tqdm(all_clean_files, desc="密度计算", unit="张"):
        img = np.array(Image.open(os.path.join("1Den_clean_cont", fname)), dtype=np.float32) / 255.0
        densities.append(compute_texture_density(img))
    densities = np.array(densities)
    density_threshold = float(np.percentile(densities, 95))
    density_info = {
        "density_threshold": density_threshold,
        "mean": float(densities.mean()),
        "std": float(densities.std()),
        "min": float(densities.min()),
        "max": float(densities.max()),
        "p95": density_threshold,
    }
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/density_threshold.json", "w") as f:
        json.dump(density_info, f, indent=2)
    print(
        f"密度阈值: {density_threshold:.4f} (训练集 95%分位) "
        f"[范围 {densities.min():.4f}~{densities.max():.4f}, "
        f"均值 {densities.mean():.4f}±{densities.std():.4f}]"
    )
    print(f"已保存到 checkpoints/density_threshold.json")

    # ════════════════════════════════════════════════════════════
    # 第二步：划分训练/验证集
    # ════════════════════════════════════════════════════════════
    total = len(all_clean_files)
    all_indices = torch.randperm(total).tolist()
    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count : train_count + val_count]

    # ════════════════════════════════════════════════════════════
    # 第三步：创建数据集
    #   训练集：启用自适应放大（密度 > P95 阈值时自动 2x 放大）
    #   验证集：不放大，保持原始分辨率评估泛化能力
    # ════════════════════════════════════════════════════════════
    train_dataset = FringeDataset(
        "1Den", "1Den_clean_cont", edge_dir="1Den_edge",
        density_threshold=density_threshold,
        safety_ratio=TRAIN_SAFETY_RATIO,
    )
    val_dataset = FringeDataset(
        "1Den", "1Den_clean_cont", edge_dir="1Den_edge",
        density_threshold=0.0,  # 0 = 不放大
    )
    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)

    # ── 统计训练集中会被放大的图像 ──
    scaled_count = 0
    scaled_levels = {}
    effective_threshold = density_threshold * TRAIN_SAFETY_RATIO
    for idx in train_indices:
        clean_path = os.path.join("1Den_clean_cont", train_dataset.files[idx])
        img = np.array(Image.open(clean_path), dtype=np.float32) / 255.0
        d = compute_texture_density(img)
        if d > effective_threshold:
            scaled_count += 1
            # 模拟放大级数
            lvl = 1
            while d > effective_threshold and lvl <= MAX_SCALE_ITER:
                lvl += 1
                img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                                 interpolation=cv2.INTER_LINEAR)
                d = compute_texture_density(img)
            scaled_levels[lvl] = scaled_levels.get(lvl, 0) + 1

    print(f"总样本库: {total} 张 | 训练: {len(train_ds)} 张 | 验证: {len(val_ds)} 张")
    print(f"边界加权系数 alpha = {edge_weight}")
    print(f"训练集自适应放大: {scaled_count}/{len(train_ds)} 张触发 (阈值>{effective_threshold:.4f})")
    if scaled_levels:
        level_desc = ", ".join(f"{lvl}级({cnt}张)" for lvl, cnt in sorted(scaled_levels.items()))
        print(f"  放大级数分布: {level_desc}  [最大 {MAX_SCALE_ITER} 级 = {2**MAX_SCALE_ITER}x]")

    # ════════════════════════════════════════════════════════════
    # 第四步：DataLoader
    #   训练集用 collate_variable_size 处理放大后的变尺寸图像
    #   验证集尺寸统一，不需要特殊 collate
    # ════════════════════════════════════════════════════════════
    num_workers = min(os.cpu_count() or 4, 4)
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        num_workers=num_workers, collate_fn=collate_variable_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=num_workers,
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")
    epochs = 30

    print(
        f"Training U-Net: {len(train_ds)} train / {len(val_ds)} val samples, {epochs} epochs"
    )

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── 训练阶段 ──
        model.train()
        train_edge_loss = 0
        train_mse_loss = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:2d}/{epochs} [Train]",
            unit="batch",
            leave=False,
        )
        for batch in train_pbar:
            noisy, clean, edge = batch
            noisy, clean, edge = noisy.to(device), clean.to(device), edge.to(device)
            optimizer.zero_grad()
            output = model(noisy)

            # 边界加权损失（主loss）+ 基础MSE（辅助监控）
            loss_edge = edge_weighted_mse_loss(output, clean, edge, alpha=edge_weight)
            loss_mse = mse_criterion(output, clean)
            loss = loss_edge

            loss.backward()
            optimizer.step()
            train_edge_loss += loss_edge.item()
            train_mse_loss += loss_mse.item()
            train_pbar.set_postfix(edge_loss=f"{loss_edge.item():.6f}")
        train_edge_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)

        # ── 验证阶段 ──
        model.eval()
        val_edge_loss = 0
        val_mse_loss = 0
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1:2d}/{epochs} [Val]  ",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for batch in val_pbar:
                noisy, clean, edge = batch
                noisy, clean, edge = noisy.to(device), clean.to(device), edge.to(device)
                output = model(noisy)

                loss_edge = edge_weighted_mse_loss(
                    output, clean, edge, alpha=edge_weight
                )
                loss_mse = mse_criterion(output, clean)

                val_edge_loss += loss_edge.item()
                val_mse_loss += loss_mse.item()
                val_pbar.set_postfix(edge_loss=f"{loss_edge.item():.6f}")
        val_edge_loss /= len(val_loader)
        val_mse_loss /= len(val_loader)
        scheduler.step()

        # 每个 epoch 保存模型
        torch.save(
            model.state_dict(), f"checkpoints/unet_denoiser_epoch_{epoch+1:02d}.pth"
        )

        is_best = ""
        if val_edge_loss < best_loss:
            best_loss = val_edge_loss
            torch.save(model.state_dict(), "checkpoints/unet_denoiser_best.pth")
            is_best = " ★ BEST"

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch+1:2d}/{epochs} | "
            f"EdgeLoss(train/val): {train_edge_loss:.6f}/{val_edge_loss:.6f} | "
            f"MSE(train/val): {train_mse_loss:.6f}/{val_mse_loss:.6f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s | Total: {total_elapsed/60:.1f}m{is_best}"
        )

    print(f"Training complete. Best val EdgeLoss: {best_loss:.6f}")
    torch.save(model.state_dict(), "checkpoints/unet_denoiser_final.pth")
    return model


if __name__ == "__main__":
    train()

