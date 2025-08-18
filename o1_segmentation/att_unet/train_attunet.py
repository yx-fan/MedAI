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
from monai.data import decollate_batch
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

# Dice metric (只统计前景)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

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
    writer.writerow(["epoch", "train_loss", "val_loss", "fg_dice", "lr"])

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
        images = batch["image"].to(device)
        masks  = batch["label"].to(device).long()   # ✅ 显式转为 long，供 DiceFocalLoss 的 to_onehot_y 使用

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, masks)          # ✅ 用 long 的 y

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

    # -------- Validation（完整替换该段）--------
    model.eval()
    val_loss = 0.0

    # 重置 metric
    dice_metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = batch["image"].to(device)
            masks  = batch["label"].to(device).long()  # ✅

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = sliding_window_inference(
                    images,
                    roi_size=(64, 64, 32) if args.debug else (128, 128, 64),
                    sw_batch_size=1 if args.debug else 2,
                    predictor=model,
                    overlap=0.25
                )
                loss = loss_fn(outputs, masks)         # ✅ 与训练一致

            val_loss += loss.item()

            # decollate -> 后处理 -> 累计到 DiceMetric
            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list      = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)

    # 聚合指标
    avg_val_loss = val_loss / max(1, len(val_loader))
    res = dice_metric.aggregate()

    # 兼容不同 MONAI 版本：可能返回 tensor 或 (tensor, not_nans)
    if isinstance(res, (tuple, list)) and len(res) == 2:
        fg_dice_tensor, not_nans = res
        # 统计有效样本数（可选）
        try:
            valid = int((not_nans if torch.is_tensor(not_nans) else torch.tensor(not_nans)).sum().item())
        except Exception:
            valid = None
    else:
        fg_dice_tensor = res
        valid = None

    # 无论是标量还是向量，都稳妥取均值再转 float
    fg_dice = float(torch.as_tensor(fg_dice_tensor).mean().item())

    # 可选：统计有效样本数（有前景的）
    try:
        valid = int(dice_metric.get_buffer("not_nans").sum().item())
    except Exception:
        valid = None

    if valid is not None:
        print(f"Val Loss: {avg_val_loss:.4f}, FG Dice={fg_dice:.4f}, (valid cases: {valid})")
    else:
        print(f"Val Loss: {avg_val_loss:.4f}, FG Dice={fg_dice:.4f}")
    log_gpu("After Validation")

    # -------- Save Logs --------
    lr_now = scheduler.get_last_lr()[0]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, fg_dice, lr_now])

    # -------- Save Models --------
    latest_path = os.path.join(save_dir, "latest_model.pth")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fg_dice": fg_dice,
    }, latest_path)

    if fg_dice > best_dice:  # 用前景 Dice 作为早停指标
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
