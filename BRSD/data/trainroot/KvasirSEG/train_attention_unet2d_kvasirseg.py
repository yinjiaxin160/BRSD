

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR

import albumentations as A
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "DATASET_B", "dataset")
sys.path.insert(0, PROJECT_ROOT)

from model.attention_unet2d import initialize_attention_unet2d_binary_b



IMAGE_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 150
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 30
DEVICE = "cuda"
SEED = 42

SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoint", "checkpoints_attention_unet2d_dataset_B")


class BinarySegDatasetB(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        if image.max() > 1:
            image = image / 255.0

        mask = (mask > 0).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        image = image.transpose(2, 0, 1)
        mask = mask[np.newaxis, :, :]

        return torch.from_numpy(image), torch.from_numpy(mask)


# ============ 增强 ============
def build_train_transform(size):
    return A.Compose([
        A.Resize(width=size, height=size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.15, rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT_101, p=0.7,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
        ], p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill=0, p=0.3,
        ),
    ])


def build_val_transform(size):
    return A.Compose([A.Resize(width=size, height=size, p=1.0)])


# ============ 损失函数 ============
class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * inter + smooth) / (union + smooth)
        return bce_loss + (1 - dice.mean())


# ============ 评估 ============
def compute_dice(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    smooth = 1e-5
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()


def compute_iou(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    smooth = 1e-5
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def tta_predict(model, images, device):
    """Test-Time Augmentation: average predictions over horizontal/vertical flips."""
    preds = []
    # Original
    logits = model(images.to(device))
    if isinstance(logits, tuple):
        logits = logits[0]
    preds.append(torch.sigmoid(logits))

    # Horizontal flip
    logits = model(torch.flip(images, dims=[3]).to(device))
    if isinstance(logits, tuple):
        logits = logits[0]
    preds.append(torch.sigmoid(torch.flip(logits, dims=[3])))

    # Vertical flip
    logits = model(torch.flip(images, dims=[2]).to(device))
    if isinstance(logits, tuple):
        logits = logits[0]
    preds.append(torch.sigmoid(torch.flip(logits, dims=[2])))

    # Average
    return torch.stack(preds).mean(dim=0)


# ============ 训练 ============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_images = np.load(os.path.join(DATA_DIR, "images_train.npy"))
    train_masks = np.load(os.path.join(DATA_DIR, "masks_train.npy"))
    val_images = np.load(os.path.join(DATA_DIR, "images_val.npy"))
    val_masks = np.load(os.path.join(DATA_DIR, "masks_val.npy"))

    print(f"train: {train_images.shape}, val: {val_images.shape}")
    print(f"device: {device}")

    train_ds = BinarySegDatasetB(train_images, train_masks, build_train_transform(IMAGE_SIZE))
    val_ds = BinarySegDatasetB(val_images, val_masks, build_val_transform(IMAGE_SIZE))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = initialize_attention_unet2d_binary_b(in_channels=3, out_channels=1).to(device)
    print(f"Attention UNet2D parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4
    )

    best_dice, best_epoch, no_improve = 0.0, 0, 0
    log_path = os.path.join(SAVE_DIR, "train.log")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()

            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True).float()
                masks = masks.to(device, non_blocking=True).float()
                probs = tta_predict(model, images, device)
                preds = (probs > 0.5).float()
                smooth = 1e-5
                inter = (preds * masks).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                val_dice += ((2 * inter + smooth) / (union + smooth)).mean().item()
                val_iou += ((inter + smooth) / (union - inter + smooth)).mean().item()

        val_dice /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))
        cur_lr = optimizer.param_groups[0]["lr"]

        msg = (f"[attention_unet2d_kvasir] Epoch [{epoch}/{EPOCHS}] "
               f"Loss: {train_loss:.4f} Val Dice: {val_dice:.4f} "
               f"Val IoU: {val_iou:.4f} LR: {cur_lr:.2e}")
        print(msg)

        with open(log_path, "a") as f:
            f.write(msg + "\n")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "latest.pth"))

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
            print(f"保存 best 模型, Dice={best_dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"早停 @ epoch {epoch}（best Dice={best_dice:.4f} @ epoch {best_epoch}）")
                break

    print(f"\n最佳 Dice={best_dice:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    train()
