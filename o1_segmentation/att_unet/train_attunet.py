import os
import csv
import time
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from monai.networks.nets import AttentionUnet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.metrics import ConfusionMatrixMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from data_loader import get_dataloaders
import wandb
import matplotlib.pyplot as plt

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

base_epochs = 100 if not args.debug else 3
num_epochs = base_epochs
start_epoch = 0

learning_rate = 2e-4
save_dir = "data/attunet_debug" if args.debug else "data/attunet"
os.makedirs(save_dir, exist_ok=True)
best_dice = -1.0

# ==============================
# WandB Init
# ==============================
wandb.init(
    project="rectal-cancer-attunet-seg",
    config={
        "epochs": num_epochs,
        "batch_size": 2 if args.debug else 4,
        "learning_rate": learning_rate,
        "architecture": "AttentionUNet"
    },
    settings=wandb.Settings(init_timeout=300, start_method="thread")
)

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
    out_channels=2,
    channels=(8, 16, 32, 64) if args.debug else (16, 32, 64, 128, 256),
    strides=(2, 2, 2) if args.debug else (2, 2, 2, 2),
).to(device)

# ==============================
# Loss, Optimizer, Scheduler
# ==============================
loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# ==============================
# Metrics
# ==============================
dice_metric = DiceMetric(include_background=False, reduction="none")
hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
precision_metric = ConfusionMatrixMetric(metric_name="precision", reduction="mean", include_background=False)
recall_metric = ConfusionMatrixMetric(metric_name="recall", reduction="mean", include_background=False)
miou_metric = ConfusionMatrixMetric(metric_name="jaccard", reduction="mean", include_background=False)
specificity_metric = ConfusionMatrixMetric(metric_name="specificity", reduction="mean", include_background=False)

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

    warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    scheduler.last_epoch = start_epoch

# ==============================
# CSV Logger
# ==============================
log_path = os.path.join(save_dir, "train_log.csv")
write_header = not (args.resume and os.path.exists(log_path))
log_mode = "a" if os.path.exists(log_path) else "w"
with open(log_path, log_mode, newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["epoch", "train_loss", "val_loss", "fg_dice_mean", "fg_dice_std",
                         "hausdorff_mean", "hausdorff_max", "precision", "recall",
                         "specificity", "miou", "lr", "grad_norm"])

# ==============================
# Helper: GPU memory logger
# ==============================
def log_gpu(stage: str):
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[GPU] {stage} | Allocated: {mem:.2f} MB")

# ==============================
# Helper: WandB prediction visualization
# ==============================
def log_prediction(image, label, pred, epoch):
    mid = image.shape[-1] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image[0, 0, :, :, mid], cmap="gray"); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(label[0, 0, :, :, mid]); plt.title("GT")
    plt.subplot(1, 3, 3); plt.imshow(pred[0, 1, :, :, mid]); plt.title("Pred")
    wandb.log({"Predictions": wandb.Image(plt)}, step=epoch)
    plt.close()

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
            loss = loss_fn(outputs, masks)
        scaler.scale(loss).backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        grad_norm = (total_norm ** 0.5)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / max(1, len(train_loader))
    print(f"Train Loss: {avg_train_loss:.4f}, GradNorm={grad_norm:.2f}")
    log_gpu("After Training")

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    dice_metric.reset()
    hausdorff_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    miou_metric.reset()
    specificity_metric.reset()

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

            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list      = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)
            hausdorff_metric(y_pred=y_pred_list, y=y_list)
            precision_metric(y_pred=y_pred_list, y=y_list)
            recall_metric(y_pred=y_pred_list, y=y_list)
            miou_metric(y_pred=y_pred_list, y=y_list)
            specificity_metric(y_pred=y_pred_list, y=y_list)

    avg_val_loss = val_loss / max(1, len(val_loader))
    dice_vals = dice_metric.aggregate()
    hausdorff_vals = hausdorff_metric.aggregate()
    fg_dice_mean = float(torch.as_tensor(dice_vals).mean().item())
    fg_dice_std = float(torch.as_tensor(dice_vals).std().item())
    hausdorff_mean = float(torch.as_tensor(hausdorff_vals).mean().item())
    hausdorff_max = float(torch.as_tensor(hausdorff_vals).max().item())
    precision_val = float(torch.as_tensor(precision_metric.aggregate()).mean().item())
    recall_val = float(torch.as_tensor(recall_metric.aggregate()).mean().item())
    miou_val = float(torch.as_tensor(miou_metric.aggregate()).mean().item())
    specificity_val = float(torch.as_tensor(specificity_metric.aggregate()).mean().item())

    print(f"Val Loss: {avg_val_loss:.4f}, Dice={fg_dice_mean:.4f}±{fg_dice_std:.4f}, "
          f"Hausdorff={hausdorff_mean:.2f} (max {hausdorff_max:.2f}), "
          f"Precision={precision_val:.4f}, Recall={recall_val:.4f}, Specificity={specificity_val:.4f}, mIoU={miou_val:.4f}")
    log_gpu("After Validation")

    if (epoch + 1) % 10 == 0:
        log_prediction(images.cpu().numpy(), masks.cpu().numpy(), outputs.cpu().numpy(), epoch+1)

    # -------- Save Logs --------
    lr_now = scheduler.get_last_lr()[0]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss,
                         fg_dice_mean, fg_dice_std,
                         hausdorff_mean, hausdorff_max,
                         precision_val, recall_val, specificity_val, miou_val, lr_now, grad_norm])

    wandb.log({
        "Loss/train": avg_train_loss,
        "Loss/val": avg_val_loss,
        "Dice/val_mean": fg_dice_mean,
        "Dice/val_std": fg_dice_std,
        "Hausdorff/val_mean": hausdorff_mean,
        "Hausdorff/val_max": hausdorff_max,
        "Precision/val": precision_val,
        "Recall/val": recall_val,
        "Specificity/val": specificity_val,
        "mIoU/val": miou_val,
        "GradNorm": grad_norm,
        "LR": lr_now
    }, step=epoch+1)

    # -------- Save Models --------
    latest_path = os.path.join(save_dir, "latest_model.pth")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fg_dice": fg_dice_mean,
    }, latest_path)

    if fg_dice_mean > best_dice:
        best_dice = fg_dice_mean
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"[INFO] Best model updated: {best_path} (FG Dice={best_dice:.4f})")

    if epoch == num_epochs - 1:
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save(model.state_dict(), final_path)
        print(f"[INFO] Final model saved: {final_path}")

    scheduler.step()

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = (num_epochs - (epoch + 1)) * epoch_time
    print(f"[ETA] Epoch time: {epoch_time/60:.2f} min | Elapsed: {elapsed/60:.2f} min | Remaining: {remaining/60:.2f} min")

wandb.finish()
