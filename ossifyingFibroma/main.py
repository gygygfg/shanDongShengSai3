"""
CNN + GAN 混合模型：基于 timm (pytorch-image-models) 的图像语义分割
- 使用 timm 预训练模型作为 CNN 编码器 (U-Net 架构)
- 对抗网络 (GAN) 提升分割精度
- 从 ../3Seg/ 原图中识别 ./3Seg_Labeled/ 中红色标记的区域
"""

import os
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE_G = 1e-4  # Generator 学习率
LEARNING_RATE_D = 4e-4  # Discriminator 学习率
IMAGE_SIZE = 256
TIMB_BACKBONE = "efficientnet_b0"  # timm 预训练 CNN 骨干
NUM_WORKERS = 2
SAVE_DIR = Path("./outputs")
SAVE_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"timm backbone: {TIMB_BACKBONE}")


# ============================================================
# 数据集：从原图和标注图中提取红色区域 mask
# ============================================================
class SegmentationDataset(Dataset):
    """加载原图 + 标注图，提取红色区域作为二值 mask"""

    def __init__(self, img_dir, label_dir, image_size=256, augment=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.augment = augment

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        self.label_paths = sorted(self.label_dir.glob("*.png"))
        assert len(self.img_paths) == len(
            self.label_paths
        ), f"图片数量不匹配: {len(self.img_paths)} vs {len(self.label_paths)}"
        print(f"加载 {len(self.img_paths)} 对图片")

        # 基础 transform
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
            ]
        )
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def extract_red_mask(label_img: Image.Image) -> np.ndarray:
        """
        从标注图中提取红色区域 -> 二值 mask (0/1)
        红色定义: R > 150, G < 80, B < 80
        """
        arr = np.array(label_img, dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mask = ((r > 150) & (g < 80) & (b < 80)).astype(np.float32)
        return mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("RGB")

        # 同步随机增强
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机旋转
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
            label = label.rotate(angle, resample=Image.BILINEAR)

        img = self.base_transform(img)
        label = self.base_transform(label)

        mask = self.extract_red_mask(label)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        img_tensor = self.to_tensor(img)  # [3, H, W]

        return img_tensor, mask


# ============================================================
# Generator: U-Net 风格，使用 timm 预训练模型作为编码器
# ============================================================
class TimmUNetGenerator(nn.Module):
    """
    基于 timm 的 U-Net 生成器（分割网络）
    编码器: timm 预训练 CNN
    解码器: 逐层上采样 + skip connection
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()

        # 使用 timm 创建编码器并获取各层通道
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,  # 返回各层特征图
        )
        # 获取各 stage 输出通道
        encoder_channels = self.encoder.feature_info.channels()
        print(f"Encoder channels: {encoder_channels}")

        # 解码器：从深层到浅层逐步上采样
        self.decoder_blocks = nn.ModuleList()
        prev_ch = encoder_channels[-1]

        for ch in reversed(encoder_channels[:-1]):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch + ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            prev_ch = ch

        # 最终输出头
        self.final_conv = nn.Sequential(
            nn.Conv2d(prev_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        # 编码
        features = self.encoder(x)  # list of feature maps [f0, f1, f2, f3, ...]

        # 解码
        x = features[-1]
        for i, block in enumerate(self.decoder_blocks):
            skip = features[-(i + 2)]  # 对应层级的 skip connection
            # 上采样到 skip 尺寸
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # 上采样到输入尺寸
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        return x  # 输出 logits，由损失函数处理 sigmoid


# ============================================================
# Discriminator: PatchGAN 判别器
# ============================================================
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN 判别器：判断分割 mask 是"真"（人工标注）还是"假"（模型生成）
    输入: 原图 + mask 拼接 [B, 4, H, W]
    """

    def __init__(self, in_channels=4, base_filters=64):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_channels, base_filters, normalize=False),  # 64
            conv_block(base_filters, base_filters * 2),  # 128
            conv_block(base_filters * 2, base_filters * 4),  # 256
            conv_block(base_filters * 4, base_filters * 8, stride=1),  # 512
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, padding=1),  # 输出单通道
        )

    def forward(self, img, mask):
        # 拼接原图和 mask
        x = torch.cat([img, mask], dim=1)  # [B, 4, H, W]
        return self.model(x)  # [B, 1, H', W']  -- PatchGAN 输出


# ============================================================
# 损失函数 (针对细/枝状结构优化)
# ============================================================
def dice_loss(pred, target, smooth=1.0):
    """Dice Loss（pred 为 sigmoid 概率）"""
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth
    )
    return (1.0 - dice).mean()


def focal_loss(pred_logits, target, alpha=0.75, gamma=2.0):
    """
    Focal Loss: 自动聚焦难分样本，对细/枝状结构更友好
    alpha: 正样本权重 (target=1 时的权重)
    gamma: 聚焦参数，越大越关注难分样本
    """
    pred_prob = torch.sigmoid(pred_logits)
    # 对正样本 (target=1): -alpha * (1-p)^gamma * log(p)
    # 对负样本 (target=0): -(1-alpha) * p^gamma * log(1-p)
    pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - pt) ** gamma
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    return (focal_weight * bce).mean()


def combined_seg_loss(pred_logits, target):
    """组合分割损失: Focal Loss + Dice Loss + 面积正则化 (适合枝状细结构)"""
    pred_prob = torch.sigmoid(pred_logits)
    # 面积正则化: L1 惩罚预测 mask 的总体积，防止"一片"
    area_reg = pred_prob.mean() * 0.1
    return (
        focal_loss(pred_logits, target, alpha=0.75, gamma=2.0)
        + dice_loss(pred_prob, target)
        + area_reg
    )


def gan_loss(pred_fake, pred_real):
    """对抗损失"""
    real_loss = F.mse_loss(pred_real, torch.ones_like(pred_real))
    fake_loss = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return (real_loss + fake_loss) * 0.5


# ============================================================
# 训练循环
# ============================================================
def train_one_epoch(
    generator, discriminator, dataloader, optimizer_g, optimizer_d, epoch
):
    generator.train()
    discriminator.train()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    g_losses, d_losses = [], []

    for imgs, masks in pbar:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        # ---- 1. 训练 Discriminator ----
        optimizer_d.zero_grad()

        with torch.no_grad():
            fake_logits = generator(imgs)
            fake_masks = torch.sigmoid(fake_logits)

        pred_fake = discriminator(imgs, fake_masks.detach())
        pred_real = discriminator(imgs, masks)

        d_loss = gan_loss(pred_fake, pred_real)
        d_loss.backward()
        optimizer_d.step()

        # ---- 2. 训练 Generator ----
        optimizer_g.zero_grad()

        fake_logits = generator(imgs)
        fake_masks = torch.sigmoid(fake_logits)

        # 分割损失 (加权 BCE + Dice)
        seg_loss = combined_seg_loss(fake_logits, masks)

        # 对抗损失
        pred_fake_for_g = discriminator(imgs, fake_masks)
        adv_loss = F.mse_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

        # 总 Generator 损失
        g_loss = seg_loss + 0.05 * adv_loss
        g_loss.backward()
        optimizer_g.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        pbar.set_postfix(
            G_loss=f"{np.mean(g_losses):.4f}",
            D_loss=f"{np.mean(d_losses):.4f}",
            Seg=f"{seg_loss.item():.4f}",
        )

    return np.mean(g_losses), np.mean(d_losses)


# ============================================================
# 验证/测试
# ============================================================
@torch.no_grad()
def evaluate(generator, dataloader):
    """评估分割 IoU 和 Dice"""
    generator.eval()
    total_iou, total_dice, count = 0.0, 0.0, 0

    for imgs, masks in dataloader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = generator(imgs)
        preds = torch.sigmoid(logits)
        preds_binary = (preds > 0.75).float()  # 与推理阈值一致

        # IoU
        intersection = (preds_binary * masks).sum(dim=(1, 2, 3))
        union = ((preds_binary + masks) > 0).float().sum(dim=(1, 2, 3))
        iou = (intersection / (union + 1e-6)).mean().item()

        # Dice
        dice = (
            (
                2
                * intersection
                / (preds_binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
            )
            .mean()
            .item()
        )

        total_iou += iou
        total_dice += dice
        count += 1

    return total_iou / count, total_dice / count


# ============================================================
# 推理：对单张图片预测并保存可视化结果
# ============================================================
def inference_and_save(generator, img_path, save_path, image_size=256):
    """对单张图片进行推理，只保存二值化之后的 mask 图像"""
    generator.eval()

    original_img = Image.open(img_path).convert("RGB")
    w, h = original_img.size

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    img_tensor = transform(original_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = torch.sigmoid(generator(img_tensor))  # [1, 1, H, W]
        pred_mask = pred_mask.squeeze().cpu().numpy()

    # 缩放回原始尺寸
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_mask_img = pred_mask_img.resize((w, h), Image.BILINEAR)

    # 二值化（用更高阈值抑制过度膨胀，枝状结构需要更精确的分割）
    pred_binary = (np.array(pred_mask_img) > 192).astype(
        np.uint8
    ) * 255  # 192/255 ≈ 0.75
    pred_binary_img = Image.fromarray(pred_binary)

    # 只保存二值化图像
    pred_binary_img.save(save_path)
    print(f"推理结果已保存: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("CNN + GAN 图像语义分割 (timm backbone)")
    print("=" * 60)

    # ---- 数据 ----
    train_dataset = SegmentationDataset(
        "../3Seg", "./3Seg_Labeled", image_size=IMAGE_SIZE, augment=True
    )
    # 小数据集: 全部用于训练，同时作为验证
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ---- 模型 ----
    generator = TimmUNetGenerator(backbone_name=TIMB_BACKBONE, pretrained=True).to(
        DEVICE
    )
    discriminator = PatchGANDiscriminator(in_channels=4).to(DEVICE)

    print(
        f"Generator 参数量: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M"
    )
    print(
        f"Discriminator 参数量: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M"
    )

    # ---- 优化器 ----
    optimizer_g = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999)
    )
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=NUM_EPOCHS)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=NUM_EPOCHS)

    # ---- 训练 ----
    best_iou = 0.0
    for epoch in range(NUM_EPOCHS):
        g_loss, d_loss = train_one_epoch(
            generator, discriminator, train_loader, optimizer_g, optimizer_d, epoch
        )
        scheduler_g.step()
        scheduler_d.step()

        # 验证
        iou, dice = evaluate(generator, val_loader)
        print(
            f"Epoch {epoch + 1:3d} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | "
            f"IoU: {iou:.4f} | Dice: {dice:.4f} | LR: {scheduler_g.get_last_lr()[0]:.2e}"
        )

        # 保存最佳模型
        if iou > best_iou:
            best_iou = iou
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "epoch": epoch,
                    "iou": iou,
                },
                SAVE_DIR / "best_model.pth",
            )
            print(f"  -> 保存最佳模型 (IoU: {best_iou:.4f})")

    print(f"\n训练完成！最佳 IoU: {best_iou:.4f}")

    # ---- 推理：对所有测试图进行预测 ----
    print("\n" + "=" * 60)
    print("推理 & 可视化")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(SAVE_DIR / "best_model.pth", map_location=DEVICE)
    print(
        f"加载最佳模型 (Epoch {checkpoint['epoch'] + 1}, IoU: {checkpoint['iou']:.4f})"
    )

    pred_dir = Path("../results")
    pred_dir.mkdir(exist_ok=True)

    for img_path in sorted(Path("../3Seg").glob("*.png")):
        save_path = pred_dir / f"3Seg_{img_path.stem}.png"
        inference_and_save(generator, img_path, save_path, IMAGE_SIZE)

    print(f"\n所有结果保存在: {pred_dir}")


if __name__ == "__main__":
    main()
