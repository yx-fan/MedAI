import os
import csv
import time
import torch
import argparse
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from data_loader import get_dataloaders
from models import (
    build_model, loss_fn,
    dice_metric, precision_metric, recall_metric, specificity_metric,
    post_pred, post_label
)
from utils import log_gpu, log_prediction

# ==============================
# Argparse
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Run in debug mode with small dataset and fewer epochs")
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (e.g., data/unet/latest_model.pth)")
parser.add_argument("--extra_epochs", type=int, default=0, help="How many more epochs to train from the loaded checkpoint")
args = parser.parse_args()

# ==============================
# Configuration
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on {device}")
cudnn.benchmark = True

base_epochs = 100 if not args.debug else 3
num_epochs = base_epochs
start_epoch = 0
learning_rate = 2e-4
save_dir = "data/unet_debug" if args.debug else "data/unet"
os.makedirs(save_dir, exist_ok=True)
best_dice = -1.0

from datetime import datetime
# ==============================
# TensorBoard Init
# ==============================
log_dir = os.path.join("tb_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logs at {log_dir}")

# ==============================
# Data Loaders
# ==============================
train_loader, val_loader = get_dataloaders(
    data_dir="./data/raw",
    batch_size=1 if args.debug else 2,
    debug=args.debug
)

# ==============================
# Model Definition
# ==============================
model = build_model(device)

# ==============================
# Loss, Optimizer, Scheduler
# ==============================
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5,
    patience=10, min_lr=1e-6, verbose=True
)

# ==============================
# AMP Scaler
# ==============================
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ==============================
# Resume (optional)
# ==============================
if args.resume:
    print(f"[INFO] Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] optimizer state load failed: {e}")
        start_epoch = int(ckpt.get("epoch", 0))
        best_dice = float(ckpt.get("fg_dice", best_dice))
        print(f"[INFO] Loaded epoch={start_epoch}, prev_fg_dice={ckpt.get('fg_dice', None)}")
    else:
        model.load_state_dict(ckpt)
        start_epoch = 0
        print("[WARN] Loaded weights only (no optimizer state)")
    total_epochs = start_epoch + max(args.extra_epochs, 0)
    if total_epochs <= start_epoch:
        total_epochs = max(base_epochs, start_epoch)
    num_epochs = total_epochs
    print(f"[INFO] Will train epochs [{start_epoch} -> {num_epochs})")

# ==============================
# CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
write_header = not (args.resume and os.path.exists(log_path))
log_mode = "a" if os.path.exists(log_path) else "w"
with open(log_path, log_mode, newline="") as f:
    writer_csv = csv.writer(f)
    if write_header:
        writer_csv.writerow([
            "epoch", "train_loss", "val_loss", "fg_dice_mean", "fg_dice_std",
            "precision", "recall", "specificity", "miou", "lr", "grad_norm"
        ])

# ==============================
# Training Loop
# ==============================
start_time = time.time()
for epoch in trange(start_epoch, num_epochs, desc="Total Progress"):
    epoch_start = time.time()
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    log_gpu("Start Epoch")

    # -------- Training --------
    model.train()
    train_loss = 0.0
    grad_norm = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].to(device)
        masks  = batch["label"].to(device).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, masks)  # --- Modified: use total_loss ---
        scaler.scale(loss).backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        grad_norm = (total_norm ** 0.5)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer); scaler.update()
        train_loss += loss.item()
    avg_train_loss = train_loss / max(1, len(train_loader))
    print(f"Train Loss: {avg_train_loss:.4f}, GradNorm={grad_norm:.2f}")
    log_gpu("After Training")

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    dice_metric.reset(); precision_metric.reset(); recall_metric.reset(); specificity_metric.reset()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = batch["image"].to(device)
            masks  = batch["label"].to(device).long()

            # --- Use sliding window inference instead of direct forward ---
            outputs = sliding_window_inference(
                images,
                roi_size=(64, 64, 32) if args.debug else (160, 160, 128),
                sw_batch_size=1 if args.debug else 4,
                predictor=model,
                overlap=0.5,          # --- Modified: increased overlap
                mode="gaussian"       # --- Modified: gaussian blending
            )
            loss = loss_fn(outputs, masks)  # --- Modified: use total_loss ---
            val_loss += loss.item()

            # --- Decollate batch before post transforms ---
            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list      = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)
            precision_metric(y_pred=y_pred_list, y=y_list)
            recall_metric(y_pred=y_pred_list, y=y_list)
            specificity_metric(y_pred=y_pred_list, y=y_list)

    avg_val_loss = val_loss / max(1, len(val_loader))
    fg_dice_mean = float(torch.as_tensor(dice_metric.aggregate()).mean().item())
    print(f"Val Loss: {avg_val_loss:.4f}, Dice={fg_dice_mean:.4f}")
    log_gpu("After Validation")

    # -------- TensorBoard log --------
    writer.add_scalar("Loss/train", avg_train_loss, epoch+1)
    writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
    writer.add_scalar("Dice/val_mean", fg_dice_mean, epoch+1)
    writer.add_scalar("GradNorm", grad_norm, epoch+1)
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("LR", current_lr, epoch+1)

    if (epoch + 1) % 10 == 0:
        log_prediction(writer, images.cpu().numpy(), masks.cpu().numpy(), outputs.cpu().numpy(), epoch+1)

    # -------- Save --------
    latest_path = os.path.join(save_dir, "latest_model.pth")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fg_dice": fg_dice_mean
    }, latest_path)
    if fg_dice_mean > best_dice:
        best_dice = fg_dice_mean
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"[INFO] Best model updated: {best_path} (FG Dice={best_dice:.4f})")

    # --- LR scheduler update ---
    scheduler.step(fg_dice_mean)

writer.close()
