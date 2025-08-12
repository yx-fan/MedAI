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


# ================== Dice 计算 ==================
def dice_score(pred, target, num_classes=2, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    if pred.ndim == 2:  # batch=1 时补维度
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    dice_list = []
    for cls in range(1, num_classes):  # 跳过背景
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice_cls = ((2. * intersection + smooth) / (union + smooth)).mean()
        dice_list.append(dice_cls.item())
    return np.mean(dice_list) if dice_list else 0.0


if __name__ == "__main__":
    # ================== 配置 ==================
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Mac 上建议 NUM_WORKERS=0
    num_workers = 0

    # ================== 数据集（npy 格式） ==================
    full_dataset = TransUNetDataset(
        img_dir=IMAGES_TR,
        img_size=IMG_SIZE,
        mode="npy",
        data_format=DATA_FORMAT
    )

    # 自动获取输入通道数
    in_ch = 1 if DATA_FORMAT == "2d" else full_dataset[0][0].shape[0]

    # 划分训练 / 验证集
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ================== 模型 ==================
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

    # ================== 训练循环 ==================
    for epoch in range(NUM_EPOCHS):
        # ---------- 训练 ----------
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

        # ---------- 验证 ----------
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

        # ---------- 日志 ----------
        log_msg = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )
        if val_auc is not None:
            log_msg += f" | Val AUC: {val_auc:.4f}"
        print(log_msg)

        # ---------- 保存最佳模型 ----------
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_path = os.path.join(
                MODEL_SAVE_DIR,
                f"transunet_best_{DATA_FORMAT}_ps{VIT_PATCH_SIZE}_d{VIT_DEPTH}_e{epoch+1}.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"✅ Best model saved to {best_path} (Dice={best_val_dice:.4f})")

    # ================== 保存最终模型 ==================
    final_path = os.path.join(
        MODEL_SAVE_DIR,
        f"transunet_final_{DATA_FORMAT}_ps{VIT_PATCH_SIZE}_d{VIT_DEPTH}.pth"
    )
    torch.save(model.state_dict(), final_path)
    print(f"🏁 Training complete. Final model saved to {final_path}")
