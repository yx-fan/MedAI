import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random

from transunet_config import *
from transunet_dataset import TransUNetDataset
from transunet_model import TransUNet
from prepare_split import prepare_split


# ================== Fix random seeds ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== Metrics ==================
def dice_score(pred, target, num_classes=2, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    if pred.ndim == 2: pred = pred.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    dice_list = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice_cls = ((2. * intersection + smooth) / (union + smooth)).mean()
        dice_list.append(dice_cls.item())
    return np.mean(dice_list) if dice_list else 0.0


def iou_score(pred, target, num_classes=2, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    if pred.ndim == 2: pred = pred.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    iou_list = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
        iou_cls = ((intersection + smooth) / (union + smooth)).mean()
        iou_list.append(iou_cls.item())
    return np.mean(iou_list) if iou_list else 0.0


# ================== Loss ==================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        pred = torch.softmax(pred, dim=1)  
        target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)


# ================== Main ==================
if __name__ == "__main__":
    set_seed(42)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(MODEL_SAVE_DIR, f"training_log_{timestamp}.txt")
    log_f = open(log_file, "w")

    # Dataset split
    prepare_split(IMAGES_TR)

    # Dataset
    train_dataset_full = TransUNetDataset(
        img_dir=IMAGES_TR,
        img_size=IMG_SIZE,
        mode="npy",
        data_format=DATA_FORMAT,
        split="train"
    )

    in_ch = 1 if DATA_FORMAT == "2d" else train_dataset_full[0][0].shape[0]

    val_size = int(len(train_dataset_full) * 0.2)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=max(1, NUM_WORKERS), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=max(1, NUM_WORKERS), pin_memory=True)

    # Model
    model = TransUNet(
        in_ch=in_ch,
        img_size=IMG_SIZE,
        patch_size=VIT_PATCH_SIZE,
        emb_size=VIT_EMBED_DIM,
        depth=VIT_DEPTH,
        heads=VIT_HEADS,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = CEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_dice = 0.0

    # Training
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                val_dice += dice_score(outputs, masks, num_classes=NUM_CLASSES)
                val_iou += iou_score(outputs, masks, num_classes=NUM_CLASSES)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        scheduler.step(avg_val_loss)

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}"
        )
        print(log_msg)
        log_f.write(log_msg + "\n")
        log_f.flush()

        # Save latest
        latest_path = os.path.join(MODEL_SAVE_DIR, "transunet_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # Save best
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_path = os.path.join(MODEL_SAVE_DIR, "transunet_best.pth")
            torch.save(model.state_dict(), best_path)
            save_msg = f"✅ Best model updated at epoch {epoch+1} (Dice={best_val_dice:.4f})"
            print(save_msg)
            log_f.write(save_msg + "\n")
            log_f.flush()

    # Save final
    final_path = os.path.join(MODEL_SAVE_DIR, "transunet_final.pth")
    torch.save(model.state_dict(), final_path)
    done_msg = f"🏁 Training complete. Final model saved to {final_path}"
    print(done_msg)
    log_f.write(done_msg + "\n")

    log_f.close()
