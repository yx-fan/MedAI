import os
import csv
import torch
import argparse
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from data_loader import get_dataloaders
from models import (
    build_model, build_loss_fn,
    dice_metric, precision_metric, recall_metric, specificity_metric,
    post_pred, post_label
)
from utils import log_prediction

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint")
parser.add_argument("--extra_epochs", type=int, default=0, help="Extra epochs from checkpoint")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on {device}")
cudnn.benchmark = True

base_epochs = 200 if not args.debug else 3
num_epochs = base_epochs
start_epoch = 0
learning_rate = 2e-4
save_dir = "data/unet_debug" if args.debug else "data/unet"
os.makedirs(save_dir, exist_ok=True)
best_dice = -1.0
USE_COMBINED_LOSS = True

log_dir = os.path.join("tb_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logs at {log_dir}")

train_loader, val_loader = get_dataloaders(
    data_dir="./data/raw",
    batch_size=1 if args.debug else 8,  # Reduced to avoid OOM
    debug=args.debug
)

model = build_model(device)
loss_fn = build_loss_fn(device, use_combined=USE_COMBINED_LOSS)
if USE_COMBINED_LOSS:
    print("[INFO] Using combined loss (DiceCE + FocalTversky + Hausdorff)")
else:
    print("[INFO] Using simple DiceCELoss")

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5,
    patience=15, min_lr=1e-6, verbose=False
)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

if args.resume:
    print(f"[INFO] Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] Optimizer state load failed: {e}")
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

log_path = os.path.join(save_dir, "train_log.csv")
write_header = not (args.resume and os.path.exists(log_path))
log_mode = "a" if os.path.exists(log_path) else "w"
log_file = open(log_path, log_mode, newline="")
writer_csv = csv.writer(log_file)
if write_header:
    writer_csv.writerow([
        "epoch", "train_loss", "val_loss", "fg_dice_mean", "fg_dice_std",
        "precision", "recall", "specificity", "lr", "grad_norm"
    ])

for epoch in trange(start_epoch, num_epochs, desc="Total Progress"):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")

    model.train()
    train_loss = 0.0
    grad_norm = 0.0
    
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        if step == len(train_loader) - 1:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            grad_norm = (total_norm ** 0.5)
        
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}, GradNorm={grad_norm:.2f}")

    model.eval()
    val_loss = 0.0
    dice_metric.reset()
    
    use_sliding_window = (epoch + 1) % 50 == 0 or epoch < 3
    compute_full_metrics = (epoch + 1) % 10 == 0
    
    if compute_full_metrics:
        precision_metric.reset()
        recall_metric.reset()
        specificity_metric.reset()
    
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["label"].to(device, non_blocking=True).long()

            if use_sliding_window:
                outputs = sliding_window_inference(
                    images,
                    roi_size=(64, 64, 32) if args.debug else (256, 256, 192),
                    sw_batch_size=1 if args.debug else 8,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )
            else:
                outputs = model(images)
            
            if compute_full_metrics:
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)
            
            if compute_full_metrics:
                precision_metric(y_pred=y_pred_list, y=y_list)
                recall_metric(y_pred=y_pred_list, y=y_list)
                specificity_metric(y_pred=y_pred_list, y=y_list)

    fg_dice_mean = float(torch.as_tensor(dice_metric.aggregate()).mean().item())
    
    if compute_full_metrics:
        avg_val_loss = val_loss / len(val_loader)
        precision = float(precision_metric.aggregate())
        recall = float(recall_metric.aggregate())
        specificity = float(specificity_metric.aggregate())
        print(f"Val Loss: {avg_val_loss:.4f}, Dice={fg_dice_mean:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
    else:
        print(f"Dice={fg_dice_mean:.4f}")

    writer.add_scalar("Loss/train", avg_train_loss, epoch+1)
    writer.add_scalar("Dice/val_mean", fg_dice_mean, epoch+1)
    writer.add_scalar("GradNorm", grad_norm, epoch+1)
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("LR", current_lr, epoch+1)
    
    if compute_full_metrics:
        writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
        writer.add_scalar("Metrics/precision", precision, epoch+1)
        writer.add_scalar("Metrics/recall", recall, epoch+1)
        writer.add_scalar("Metrics/specificity", specificity, epoch+1)

    if (epoch + 1) % 20 == 0:
        log_prediction(writer, images.cpu().numpy(), masks.cpu().numpy(), outputs.cpu().numpy(), epoch+1)

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
        print(f"[INFO] Best model updated: Dice={best_dice:.4f}")

    if compute_full_metrics:
        writer_csv.writerow([
            epoch + 1, avg_train_loss, avg_val_loss, fg_dice_mean, 0.0,
            precision, recall, specificity, current_lr, grad_norm
        ])
        log_file.flush()
    else:
        writer_csv.writerow([
            epoch + 1, avg_train_loss, 0.0, fg_dice_mean, 0.0,
            0.0, 0.0, 0.0, current_lr, grad_norm
        ])

    scheduler.step(fg_dice_mean)

log_file.close()
writer.close()
