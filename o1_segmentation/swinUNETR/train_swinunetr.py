import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm, trange

from data_loader import get_dataloaders  # make sure this exists

# ==============================
# Configuration
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # avoid using MPS
print(f"[INFO] Training on {device}")

num_epochs = 50
learning_rate = 1e-4
save_dir = "data/swinunetr"
os.makedirs(save_dir, exist_ok=True)

# Track best validation Dice
best_dice = -1.0  

# ==============================
# Data Loaders
# ==============================
train_loader, val_loader = get_dataloaders(data_dir="./data/raw", batch_size=2)

# ==============================
# Model Definition
# ==============================
model = SwinUNETR(
    in_channels=1,
    out_channels=2,  # background + foreground
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
# Init CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "dice_score"])

# ==============================
# Training Loop
# ==============================
start_time = time.time()

for epoch in trange(num_epochs, desc="Total Progress"):  # ✅ main progress bar
    epoch_start = time.time()
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")

    # -------- Training --------
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        images, masks = batch["image"].to(device), batch["label"].to(device)

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
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images, masks = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            dice_metric(y_pred=outputs, y=masks)

    avg_val_loss = val_loss / len(val_loader)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Val Loss: {avg_val_loss:.4f}, Dice: {dice_score:.4f}")

    # -------- Save Logs --------
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, dice_score])

    # -------- Save Checkpoints --------
    latest_path = os.path.join(save_dir, "latest_model.pth")
    torch.save(model.state_dict(), latest_path)

    if dice_score > best_dice:
        best_dice = dice_score
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"[INFO] Best model updated: {best_path} (Dice={best_dice:.4f})")

    if epoch == num_epochs - 1:
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save(model.state_dict(), final_path)
        print(f"[INFO] Final model saved: {final_path}")

    # -------- ETA Estimation --------
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = (num_epochs - (epoch + 1)) * epoch_time
    print(f"[ETA] Epoch time: {epoch_time/60:.2f} min | "
          f"Elapsed: {elapsed/60:.2f} min | "
          f"Remaining: {remaining/60:.2f} min")
