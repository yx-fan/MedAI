import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

from transunet_config import *
from transunet_dataset import TransUNetDataset
from transunet_model import TransUNet


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
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Create log file
    log_file = os.path.join(MODEL_SAVE_DIR, "training_log.txt")
    log_f = open(log_file, "w")

    # ================== Dataset (npy format) ==================
    full_dataset = TransUNetDataset(
        img_dir=IMAGES_TR,
        img_size=IMG_SIZE,
        mode="npy",
        data_format=DATA_FORMAT
    )

    # Automatically infer input channel size
    in_ch = 1 if DATA_FORMAT == "2d" else full_dataset[0][0].shape[0]

    # Split dataset into training / validation sets
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

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
        log_f.write(log_msg + "\n")   # save to file
        log_f.flush()

        # ---------- Save best model ----------
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_path = os.path.join(
                MODEL_SAVE_DIR,
                f"transunet_best_{DATA_FORMAT}_ps{VIT_PATCH_SIZE}_d{VIT_DEPTH}_e{epoch+1}.pth"
            )
            torch.save(model.state_dict(), best_path)
            save_msg = f"✅ Best model saved to {best_path} (Dice={best_val_dice:.4f})"
            print(save_msg)
            log_f.write(save_msg + "\n")
            log_f.flush()

    # ================== Save final model ==================
    final_path = os.path.join(
        MODEL_SAVE_DIR,
        f"transunet_final_{DATA_FORMAT}_ps{VIT_PATCH_SIZE}_d{VIT_DEPTH}.pth"
    )
    torch.save(model.state_dict(), final_path)
    done_msg = f"🏁 Training complete. Final model saved to {final_path}"
    print(done_msg)
    log_f.write(done_msg + "\n")

    # Close log file
    log_f.close()
