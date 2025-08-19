import os
import csv
import time
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, TverskyLoss
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
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (e.g., data/unet/latest_model.pth)")
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
save_dir = "data/unet_debug" if args.debug else "data/unet"
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
# Model Definition (UNet)
# ==============================
channels = (8, 16, 32, 64) if args.debug else (16, 32, 64, 128, 256)
strides  = (2, 2, 2)        if args.debug else (2, 2, 2, 2)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=channels,
    strides=strides,
    num_res_units=3,
    norm="instance",
    act="PRELU",
    dropout=0.0
).to(device)

# ==============================
# Losses（课程式：前期 DiceCE，后期 Tversky + DiceCE）
# ==============================
# 小目标友好：Tversky（偏向召回，beta>alpha）
loss_tv = TverskyLoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    alpha=0.3,   # FP 权重
    beta=0.7     # FN 权重（更重）
)
# 稳定语义与边界：Dice + 加权 CE
ce_weight = torch.tensor([0.2, 0.8], device=device)  # [bg, fg]
loss_dicece = DiceCELoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,   # dice 部分的权重
    lambda_ce=1.0,     # ce 部分的权重
)
CURRICULUM_EPOCHS = 15  # 前 15 个“当前会话 epoch”用 DiceCE 热身

# ==============================
# Optimizer & Scheduler
# ==============================
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
scheduler.last_epoch = -1  # 新训练时从 -1 开始，下一次 step 即 0

# Dice metric（忽略背景）
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

# ==============================
# AMP (混合精度)
# ==============================
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ==============================
# 输出层前景先验偏置（仅非 resume）
# ==============================
def init_pos_logit_bias(model, pos_prior=1e-3):
    """将最终前景通道的 bias 设为 ln(p/(1-p))，加速极小前景收敛。"""
    with torch.no_grad():
        bias_val = float(torch.log(torch.tensor(pos_prior/(1-pos_prior))))
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv3d) and m.out_channels == 2:
                last_conv = m
        if last_conv is not None:
            if last_conv.bias is None:
                last_conv.bias = torch.nn.Parameter(torch.zeros(last_conv.out_channels, device=next(model.parameters()).device))
            # 背景通道保持 0，前景通道给负偏置
            last_conv.bias[1].fill_(bias_val)

# ==============================
# Resume（并对齐 scheduler）
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

    # 重建 scheduler，用总轮数对齐，并设置 last_epoch=start_epoch-1
    warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    scheduler.last_epoch = start_epoch - 1
else:
    init_pos_logit_bias(model, pos_prior=1e-3)
    num_epochs = base_epochs
    start_epoch = 0

# ==============================
# CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
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
        masks  = batch["label"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            # 课程式损失：当前会话相对 epoch
            session_epoch = epoch - start_epoch
            if session_epoch < CURRICULUM_EPOCHS:
                loss = loss_dicece(outputs, masks)
            else:
                loss = 0.7 * loss_tv(outputs, masks) + 0.3 * loss_dicece(outputs, masks)

        # 梯度裁剪（AMP：先 unscale 再 clip）
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if step < 3:
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
                    roi_size=(160, 160, 96) if not args.debug else (48, 48, 24),
                    sw_batch_size=1 if args.debug else 2,
                    predictor=model,
                    overlap=0.5
                )
                # 验证端沿用当前损失（仅监控）
                session_epoch = epoch - start_epoch
                if session_epoch < CURRICULUM_EPOCHS:
                    loss = loss_dicece(outputs, masks)
                else:
                    loss = 0.7 * loss_tv(outputs, masks) + 0.3 * loss_dicece(outputs, masks)

            val_loss += loss.item()

            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list      = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)

    avg_val_loss = val_loss / max(1, len(val_loader))
    res = dice_metric.aggregate()

    if isinstance(res, (tuple, list)) and len(res) == 2:
        fg_dice_tensor, _ = res
    else:
        fg_dice_tensor = res

    fg_dice = float(torch.as_tensor(fg_dice_tensor).mean().item())

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
