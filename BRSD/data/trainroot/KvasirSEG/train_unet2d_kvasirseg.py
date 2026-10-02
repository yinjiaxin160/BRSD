

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "DATASET_B", "dataset")
sys.path.insert(0, PROJECT_ROOT)

from model.unet2d import initialize_unet2d_binary


# IMAGE_SIZE = 256
IMAGE_SIZE = 384
# BATCH_SIZE = 8 
BATCH_SIZE = 4       
NUM_WORKERS = 4
EPOCHS = 150
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 30
DEVICE = "cuda"
SEED = 42

SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoint", "checkpoints_unet2d_dataset_B")

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


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels, per_image=True):
    if per_image:
        loss = sum(lovasz_hinge_flat(
            logits[i].view(-1), labels[i].view(-1))
            for i in range(len(logits))) / len(logits)
    else:
        loss = lovasz_hinge_flat(logits.view(-1), labels.view(-1))
    return loss


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):

    def __init__(self, tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.lovasz_weight = 0.0  

    def set_lovasz_weight(self, w):
        self.lovasz_weight = w

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        tvk = self.tversky(logits, targets)
        total = bce + tvk
        if self.lovasz_weight > 0:
            total = total + self.lovasz_weight * lovasz_hinge(logits, targets, per_image=True)
        return total



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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载 npy
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

 
    model = initialize_unet2d_binary(
        in_channels=3, out_channels=1,
        features=(64, 128, 256, 512, 512),
    ).to(device)
    print(f"UNet2D parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(tversky_alpha=0.7, tversky_beta=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 5
    def warmup_lambda(epoch):
        if epoch <= warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)

    best_iou, best_epoch, no_improve = 0.0, 0, 0
    log_path = os.path.join(SAVE_DIR, "train.log")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
       
        lovasz_w = max(0.0, min(0.5, (epoch - 30) * 0.05))
        criterion.set_lovasz_weight(lovasz_w)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True).float()
                masks = masks.to(device, non_blocking=True).float()
                logits = model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                val_dice += compute_dice(logits, masks)
                val_iou += compute_iou(logits, masks)

        val_dice /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))
        cur_lr = optimizer.param_groups[0]["lr"]

        msg = (f"[unet2d_kvasir] Epoch [{epoch}/{EPOCHS}] "
               f"Loss: {train_loss:.4f} Val Dice: {val_dice:.4f} "
               f"Val IoU: {val_iou:.4f} LR: {cur_lr:.2e}")
        print(msg)

        with open(log_path, "a") as f:
            f.write(msg + "\n")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "latest.pth"))

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
            print(f" 保存 best 模型, IoU={best_iou:.4f} Dice={val_dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"早停 @ epoch {epoch}（best IoU={best_iou:.4f} @ epoch {best_epoch}）")
                break

    print(f"\n最佳 IoU={best_iou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    train()
