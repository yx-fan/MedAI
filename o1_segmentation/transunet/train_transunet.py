import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from datetime import datetime
import random

from transunet_config import *
from transunet_dataset import TransUNetDataset
from transunet_model import TransUNet
from prepare_split import prepare_split


# ================== Fix random seeds for reproducibility ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== Dice Calculation ==================
def dice_score(pred, target, num_classes=2, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)  # convert logits to predicted classes [B, H, W]
    if pred.ndim == 2:  # add batch dimension if missing
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    dice_list = []
    for cls in range(1, num_classes):  # skip background class
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice_cls = ((2. * intersection + smooth) / (union + smooth)).mean()
        dice_list.append(dice_cls.item())
    return np.mean(dice_list) if dice_list else 0.0


if __name__ == "__main__":
    # ================== Configuration ==================
    set_seed(42)  # ensure reproducibility

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(MODEL_SAVE_DIR, f"training_log_{timestamp}.txt")
    log_f = open(log_file, "w")

    # ================== Prepare dataset split ==================
    prepare_split(IMAGES_TR)  # will update meta.csv if needed

    # ================== Dataset (npy format) ==================
    train_dataset_full = TransUNetDataset(
        img_dir=IMAGES_TR,
        img_size=IMG_SIZE,
        mode="npy",
        data_format=DATA_FORMAT,
        split="train"
    )

    # automatically infer input channels
    in_ch = 1 if DATA_FORMAT == "2d" else train_dataset_full[0][0].shape[0]

    # split train / val
    val_size = int(len(train_dataset_full) * 0.2)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ================== Model ==================
    model = TransUNet(
        in_ch=in_ch,
        img_size=IMG_SIZE,
        patch_size=VIT_PATCH_SIZE,
        emb_size=VIT_EMBED_DIM,
        depth=VIT_DEPTH,
        heads=VIT_HEADS,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    best_val_dice = 0.0

    # ================== Training Loop ==================
    for epoch in range(NUM_EPOCHS):
        # ---------- Training ----------
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

        # ---------- Validation ----------
        model.eval()
        val_loss = 0
        val_dice = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                val_dice += dice_score(outputs, masks, num_classes=NUM_CLASSES)

                # For binary classification: collect predictions for AUC
                if NUM_CLASSES == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
                    targets = masks.cpu().numpy().flatten()
                    all_preds.extend(probs)
                    all_targets.extend(targets)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        # update scheduler
        scheduler.step(avg_val_loss)

        val_auc = None
        if NUM_CLASSES == 2 and all_targets:
            try:
                val_auc = roc_auc_score(all_targets, all_preds)
            except ValueError:
                val_auc = None

        # ---------- Logging ----------
        log_msg = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )
        if val_auc is not None:
            log_msg += f" | Val AUC: {val_auc:.4f}"
        print(log_msg)
        log_f.write(log_msg + "\n")
        log_f.flush()

        # ---------- Save latest model (always overwrite) ----------
        latest_path = os.path.join(MODEL_SAVE_DIR, "transunet_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # ---------- Save best model (overwrite only if improved) ----------
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_path = os.path.join(MODEL_SAVE_DIR, "transunet_best.pth")
            torch.save(model.state_dict(), best_path)
            save_msg = f"✅ Best model updated at epoch {epoch+1} (Dice={best_val_dice:.4f})"
            print(save_msg)
            log_f.write(save_msg + "\n")
            log_f.flush()

    # ================== Save final model ==================
    final_path = os.path.join(MODEL_SAVE_DIR, "transunet_final.pth")
    torch.save(model.state_dict(), final_path)
    done_msg = f"🏁 Training complete. Final model saved to {final_path}"
    print(done_msg)
    log_f.write(done_msg + "\n")

    # Close log file
    log_f.close()
