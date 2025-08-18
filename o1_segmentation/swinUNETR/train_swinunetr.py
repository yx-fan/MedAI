import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm

from data_loader import get_dataloaders  # 你之前写的data_loader.py里提供dataloader

# ==============================
# 配置
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 不要用 mps
print(f"[INFO] Training on {device}")

num_epochs = 50
learning_rate = 1e-4

# ==============================
# 数据加载
# ==============================
train_loader, val_loader = get_dataloaders(data_dir="./data/raw", batch_size=2)  # 调整路径确认！

# ==============================
# 模型定义
# ==============================
model = SwinUNETR(
    in_channels=1,
    out_channels=2,  # 背景 + 前景
    feature_size=48,
    use_checkpoint=True,
).to(device)

# ==============================
# Loss, Optimizer, Metrics
# ==============================
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ==============================
# 训练循环
# ==============================
for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")

    # -------- Train --------
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            dice_metric(y_pred=outputs, y=masks)

    avg_val_loss = val_loss / len(val_loader)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Val Loss: {avg_val_loss:.4f}, Dice: {dice_score:.4f}")

    # -------- Save checkpoint --------
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")
    print(f"Checkpoint saved: checkpoint_epoch{epoch+1}.pth")
