import os
import csv
import time
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models.segmentation import deeplabv3_resnet50

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from data_loader_2d import get_dataloaders   # 👈 我会写一个2D版 dataloader
import wandb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ==============================
# Post transforms
# ==============================
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)

# ==============================
# Argparse
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--resume", type=str, default="", help="Resume checkpoint")
parser.add_argument("--extra_epochs", type=int, default=0)
args = parser.parse_args()

# ==============================
# Config
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
print(f"[INFO] Training on {device}")

base_epochs = 100 if not args.debug else 3
num_epochs = base_epochs
start_epoch = 0
learning_rate = 1e-4
save_dir = "data/deeplab_debug" if args.debug else "data/deeplab"
os.makedirs(save_dir, exist_ok=True)
best_dice = -1.0

# ==============================
# WandB & TB
# ==============================
wandb.init(
    project="rectal-cancer-deeplab",
    config={"epochs": num_epochs, "batch_size": 4, "lr": learning_rate, "arch": "DeepLabV3+"},
    settings=wandb.Settings(init_timeout=300, start_method="thread")
)
log_dir = os.path.join("tb_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

# ==============================
# Data Loaders
# ==============================
train_loader, val_loader = get_dataloaders(
    data_dir="./data/raw", batch_size=4 if not args.debug else 2, debug=args.debug
)

# ==============================
# Model
# ==============================
model = deeplabv3_resnet50(pretrained=False, num_classes=2)
# 修改输入通道
old_conv = model.backbone.conv1
model.backbone.conv1 = torch.nn.Conv2d(
    in_channels=1,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)
model = model.to(device)

# ==============================
# Loss, Optimizer, Scheduler
# ==============================
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# ==============================
# Metrics
# ==============================
dice_metric = DiceMetric(include_background=False, reduction="none")
precision_metric = ConfusionMatrixMetric("precision", reduction="mean", include_background=False)
recall_metric = ConfusionMatrixMetric("recall", reduction="mean", include_background=False)
specificity_metric = ConfusionMatrixMetric("specificity", reduction="mean", include_background=False)

# ==============================
# AMP Scaler
# ==============================
scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

# ==============================
# Training Loop
# ==============================
for epoch in trange(start_epoch, num_epochs, desc="Total Progress"):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].to(device)   # [B, 1, H, W]
        masks  = batch["label"].to(device).long() # [B, H, W]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            outputs = model(images)["out"]   # DeepLabV3+ 返回dict
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / max(1, len(train_loader))

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    dice_metric.reset(); precision_metric.reset(); recall_metric.reset(); specificity_metric.reset()
    ious = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks  = batch["label"].to(device).long()

            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                outputs = model(images)["out"]
                loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            y_pred_list = [post_pred(o) for o in decollate_batch(outputs)]
            y_list      = [post_label(y) for y in decollate_batch(masks)]
            dice_metric(y_pred=y_pred_list, y=y_list)
            precision_metric(y_pred=y_pred_list, y=y_list)
            recall_metric(y_pred=y_pred_list, y=y_list)
            specificity_metric(y_pred=y_pred_list, y=y_list)

            for p, t in zip(y_pred_list, y_list):
                pred_bin = (p[:,1]>0.5).flatten().int()
                true_bin = t[:,1].flatten().int()
                inter = (pred_bin & true_bin).sum().float()
                union = (pred_bin | true_bin).sum().float()
                if union>0: ious.append((inter/union).item())

    avg_val_loss = val_loss / max(1, len(val_loader))
    fg_dice = float(torch.as_tensor(dice_metric.aggregate()).mean().item())
    precision_val = float(torch.as_tensor(precision_metric.aggregate()).mean().item())
    recall_val = float(torch.as_tensor(recall_metric.aggregate()).mean().item())
    specificity_val = float(torch.as_tensor(specificity_metric.aggregate()).mean().item())
    miou_val = float(torch.tensor(ious).mean().item()) if ious else 0.0

    print(f"Val Loss: {avg_val_loss:.4f}, Dice={fg_dice:.4f}, Precision={precision_val:.4f}, Recall={recall_val:.4f}, mIoU={miou_val:.4f}")

    wandb.log({
        "Loss/train": avg_train_loss, "Loss/val": avg_val_loss,
        "Dice/val": fg_dice, "Precision/val": precision_val, "Recall/val": recall_val,
        "Specificity/val": specificity_val, "mIoU/val": miou_val
    }, step=epoch+1)

    if fg_dice > best_dice:
        best_dice = fg_dice
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    scheduler.step()

wandb.finish()
writer.close()
