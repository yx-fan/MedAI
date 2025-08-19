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
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (e.g., data/attunet/latest_model.pth)")
parser.add_argument("--extra_epochs", type=int, default=0, help="How many more epochs to train from the loaded checkpoint")
args = parser.parse_args()

# ==============================
# Configuration
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on {device}")
cudnn.benchmark = True

base_epochs = 50 if not args.debug else 3
num_epochs = base_epochs
start_epoch = 0

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

# 初始（若 resume 会在下面重建并对齐）
warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# Dice metric（忽略背景）
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

# ==============================
# AMP (混合精度)
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
        # 兼容只含权重的 best_model.pth
        model.load_state_dict(ckpt)
        start_epoch = 0
        print("[WARN] Loaded weights only (no optimizer state)")

    # 续训“总轮数”= 已训 + 追加；若未传 extra，则至少不小于已训
    total_epochs = start_epoch + max(args.extra_epochs, 0)
    if total_epochs <= start_epoch:
        total_epochs = max(base_epochs, start_epoch)
    num_epochs = total_epochs
    print(f"[INFO] Will train epochs [{start_epoch} -> {num_epochs})")

    # 依据新的总轮数重建 scheduler，并将 last_epoch 对齐
    warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    scheduler.last_epoch = start_epoch
else:
    num_epochs = base_epochs
    start_epoch = 0

# ==============================
# CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
# 如果是全新训练或无日志文件 -> 写表头；续训且已有日志 -> 追加不写表头
write_header = not (args.resume and os.path.exists(log_path))
log_mode = "a" if os.path.exists(log_path) else "w"
with open(log_path, log_mode, newline="") as f:
    writer = csv.writer(f)
    if write_header:
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
for epoch in trange(start_epoch, num_epochs, desc="Total Progress"):
    epoch_start = time.time()
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    log_gpu("Start Epoch")

    # -------- Training --------
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].to(device)
        masks  = batch["label"].to(device).long()  # 确保整型类标

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if step < 3:  # 打印前3个 batch 的 loss 与显存
            print(f"[Train][Batch {step}] Loss: {loss.item():.4f}")
            log_gpu(f"After Batch {step}")

    avg_train_loss = train_loss / max(1, len(train_loader))
    print(f"Train Loss: {avg_train_loss:.4f}")
    log_gpu("After Training")

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    dice_metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = batch["image"].to(device)
            masks  = batch["label"].to(device).long()

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

            # decollate -> 后处理 -> 累计到 DiceMetric
            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list      = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)

    avg_val_loss = val_loss / max(1, len(val_loader))
    res = dice_metric.aggregate()

    # 兼容不同 MONAI 版本：可能返回 tensor 或 (tensor, not_nans)
    if isinstance(res, (tuple, list)) and len(res) == 2:
        fg_dice_tensor, not_nans = res
    else:
        fg_dice_tensor = res

    # 稳妥取均值为标量
    fg_dice = float(torch.as_tensor(fg_dice_tensor).mean().item())

    # 有效样本数（有前景的）
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
    # 注意：get_last_lr 在 PyTorch 中是 list
    lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
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

    if fg_dice > best_dice:
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
