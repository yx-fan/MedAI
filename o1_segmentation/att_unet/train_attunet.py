import os
import csv
import time
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from monai.networks.nets import AttentionUnet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from data_loader import get_dataloaders

# ==============================
# Post transforms
# ==============================
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)

# ==============================
# Argparse
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Run in debug mode with small dataset and fewer epochs")
args = parser.parse_args()

# ==============================
# Configuration
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on {device}")
cudnn.benchmark = True

num_epochs = 50 if not args.debug else 3
learning_rate = 1e-4
save_dir = "data/attunet_debug" if args.debug else "data/attunet"
os.makedirs(save_dir, exist_ok=True)
best_dice = -1.0  

# ==============================
# Data Loaders
# ==============================
train_loader, val_loader = get_dataloaders(
    data_dir="./data/raw",
    batch_size=2 if args.debug else 4,
    debug=args.debug
)

# ==============================
# Model Definition
# ==============================
model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,  # background + foreground
    channels=(8, 16, 32, 64) if args.debug else (16, 32, 64, 128, 256),
    strides=(2, 2, 2) if args.debug else (2, 2, 2, 2),
).to(device)

# ==============================
# Loss, Optimizer, Scheduler
# ==============================
loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# warmup + cosine schedule
warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# Dice metric (per-class, accumulate over val set)
dice_metric = DiceMetric(include_background=True, reduction="none")

# ==============================
# AMP (混合精度)
# ==============================
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ==============================
# CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "fg_dice", "bg_dice", "lr"])

# ==============================
# Helper: GPU memory logger
# ==============================
def log_gpu(stage: str):
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[GPU] {stage} | Allocated: {mem:.2f} MB")

# ==============================
# Training Loop
# ==============================
start_time = time.time()
for epoch in trange(num_epochs, desc="Total Progress"):
    epoch_start = time.time()
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    log_gpu("Start Epoch")

    # -------- Training --------
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images, masks = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if step < 3:  # 打印前3个 batch
            print(f"[Train][Batch {step}] Loss: {loss.item():.4f}")
            log_gpu(f"After Batch {step}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")
    log_gpu("After Training")

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    num_fg_cases = 0  # 有前景的样本数
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images, masks = batch["image"].to(device), batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = sliding_window_inference(
                    images,
                    roi_size=(64, 64, 32) if args.debug else (128, 128, 64),
                    sw_batch_size=1 if args.debug else 2,
                    predictor=model,
                    overlap=0.25
                )
                loss = loss_fn(outputs, masks)

            val_loss += loss.item()

            # 后处理
            outputs = post_pred(outputs).cpu()
            masks = post_label(masks).cpu()

            # 判断是否有前景
            if masks.max() > 0:
                dice_metric(y_pred=outputs, y=masks)
                num_fg_cases += 1

            # 打印部分 batch Dice
            if step < 2:
                tmp_metric = DiceMetric(include_background=True, reduction="none")
                if masks.max() > 0:
                    tmp_metric(y_pred=outputs, y=masks)
                    dice_values = tmp_metric.aggregate().cpu().numpy().tolist()
                    tmp_metric.reset()
                    print(f"[Val][Batch {step}] Dice per class [bg, fg]: {dice_values}")
                else:
                    print(f"[Val][Batch {step}] skipped (no foreground in GT)")

    avg_val_loss = val_loss / len(val_loader)

    # 最终 Dice（只统计有前景的样本）
    dice_scores = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()

    # 兼容 batch 输出
    if dice_scores.ndim == 2:
        dice_scores = dice_scores.mean(axis=0)

    bg_dice, fg_dice = float(dice_scores[0]), float(dice_scores[1])
    mean_dice = (bg_dice + fg_dice) / 2.0

    print(f"Val Loss: {avg_val_loss:.4f}, "
        f"FG Dice={fg_dice:.4f}, BG Dice={bg_dice:.4f}, Mean Dice={mean_dice:.4f}, "
        f"(FG cases used: {num_fg_cases})")
    log_gpu("After Validation")

    # -------- Save Logs --------
    lr_now = scheduler.get_last_lr()[0]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, fg_dice, bg_dice, mean_dice, lr_now])

    # -------- Save Models --------
    latest_path = os.path.join(save_dir, "latest_model.pth")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fg_dice": fg_dice,
        "bg_dice": bg_dice,
    }, latest_path)

    if fg_dice > best_dice:  # 用前景 Dice 作为 early stopping
        best_dice = fg_dice
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"[INFO] Best model updated: {best_path} (FG Dice={best_dice:.4f})")

    if epoch == num_epochs - 1:
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save(model.state_dict(), final_path)
        print(f"[INFO] Final model saved: {final_path}")

    # -------- Scheduler Step --------
    scheduler.step()

    # -------- ETA --------
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = (num_epochs - (epoch + 1)) * epoch_time
    print(f"[ETA] Epoch time: {epoch_time/60:.2f} min | Elapsed: {elapsed/60:.2f} min | Remaining: {remaining/60:.2f} min")
