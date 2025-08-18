# train_swinunetr.py
import os
import csv
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm, trange
from data_loader import get_dataloaders

# ==============================
# Config
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on {device}")

num_epochs = 100
learning_rate = 1e-4
save_dir = "data/swinunetr"
os.makedirs(save_dir, exist_ok=True)
best_dice = -1.0  

# ==============================
# Data
# ==============================
train_loader, val_loader = get_dataloaders(
    data_dir="./data/raw",
    batch_size=2,
    patch_size=(160, 160, 64)
)

# ==============================
# Model (MONAI 1.5 不要 img_size)
# ==============================
model = SwinUNETR(
    in_channels=1,
    out_channels=2,   # 背景 + 前景
    feature_size=48,
    use_checkpoint=True,
).to(device)

# ==============================
# Loss / Optimizer / Scheduler / Metrics
# ==============================
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

# ✅ 混合精度工具 (新写法)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ==============================
# CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "dice_score", "lr"])

# ==============================
# Training Loop
# ==============================
start_time = time.time()
for epoch in trange(num_epochs, desc="Total Progress"):
    epoch_start = time.time()
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")

    # ---- Training ----
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        images, masks = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images, masks = batch["image"].to(device), batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = sliding_window_inference(
                    images, roi_size=(160, 160, 64),
                    sw_batch_size=1, predictor=model, overlap=0.25
                )
                loss = loss_fn(outputs, masks)

            val_loss += loss.item()
            dice_metric(y_pred=outputs, y=masks)

    avg_val_loss = val_loss / len(val_loader)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Val Loss: {avg_val_loss:.4f}, Dice: {dice_score:.4f}")

    # ---- Save Logs ----
    lr_now = scheduler.get_last_lr()[0]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, dice_score, lr_now])

    # ---- Save Models ----
    latest_path = os.path.join(save_dir, "latest_model.pth")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dice": dice_score,
    }, latest_path)

    if dice_score > best_dice:
        best_dice = dice_score
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"[INFO] Best model updated: {best_path} (Dice={best_dice:.4f})")

    if epoch == num_epochs - 1:
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save(model.state_dict(), final_path)
        print(f"[INFO] Final model saved: {final_path}")

    # ---- Scheduler Step ----
    scheduler.step()

    # ---- ETA ----
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = (num_epochs - (epoch + 1)) * epoch_time
    print(f"[ETA] {epoch_time/60:.2f} min | Elapsed {elapsed/60:.2f} min | Remaining {remaining/60:.2f} min")
