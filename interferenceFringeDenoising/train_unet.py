"""
在现有条纹数据上训练 U-Net 去噪模型（1Den -> 1Den_clean）
"""
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
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
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)

# ── 数据集 ────────────────────────────────────────────────────
class FringeDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(noisy_dir))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy = np.array(Image.open(os.path.join(self.noisy_dir, fname)), dtype=np.float32) / 255.0
        clean = np.array(Image.open(os.path.join(self.clean_dir, fname)), dtype=np.float32) / 255.0
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        clean = torch.from_numpy(clean).unsqueeze(0)
        return noisy, clean

# ── 训练 ───────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 动态获取系统并发数 ──
    cpu_count = os.cpu_count() or 4
    if device.type == "cpu":
        # 显式设置 PyTorch 线程数 = 物理核心数
        torch.set_num_threads(cpu_count)
        orig_threads = torch.get_num_threads()
        print(f"CPU cores detected: {cpu_count}, PyTorch threads set to: {orig_threads}")
    else:
        orig_threads = None
        print(f"CPU cores detected: {cpu_count} (using GPU, CPU threads not throttled)")

    model = UNet(in_ch=1, out_ch=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    dataset = FringeDataset("1Den", "1Den_clean")
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    # 动态 DataLoader 并发：num_workers = 核心数（上限 8，避免 I/O 过载）
    num_workers_full = min(cpu_count, 8)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=num_workers_full)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=num_workers_full)

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    epochs = 60
    threshold_epoch = int(epochs * 0.9)  # 54

    print(f"Training U-Net: {n_train} train / {n_val} val samples, {epochs} epochs")
    print(f"  Full cores ({num_workers_full} workers / {orig_threads or 'N/A'} threads) for first {threshold_epoch} epochs")
    print(f"  Half cores for last {epochs - threshold_epoch} epochs")

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── 90% 后降到一半核心 ──
        if epoch == threshold_epoch and device.type == "cpu":
            half_threads = max(1, orig_threads // 2)
            torch.set_num_threads(half_threads)
            half_workers = max(1, num_workers_full // 2)
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=half_workers)
            val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=half_workers)
            print(f"  [Epoch {epoch+1}] Reducing cores: threads={half_threads}, workers={half_workers}")

        # ── 训练阶段 ──
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs} [Train]", unit="batch", leave=False)
        for noisy, clean in train_pbar:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss /= len(train_loader)

        # ── 验证阶段 ──
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:2d}/{epochs} [Val]  ", unit="batch", leave=False)
        with torch.no_grad():
            for noisy, clean in val_pbar:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                batch_loss = criterion(output, clean).item()
                val_loss += batch_loss
                val_pbar.set_postfix(loss=f"{batch_loss:.6f}")
        val_loss /= len(val_loader)
        scheduler.step()

        # 每个 epoch 都保存
        epoch_path = f"checkpoints/unet_denoiser_epoch_{epoch+1:02d}.pth"
        torch.save(model.state_dict(), epoch_path)

        is_best = ""
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/unet_denoiser_best.pth")
            is_best = " ★ BEST"

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.6f} | {elapsed:.1f}s | Total: {total_elapsed/60:.1f}m{is_best}")

    print(f"Training complete. Best val loss: {best_loss:.6f}")
    torch.save(model.state_dict(), "checkpoints/unet_denoiser_final.pth")
    return model


if __name__ == "__main__":
    train()
