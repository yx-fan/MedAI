import os
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models.segmentation import deeplabv3_resnet50

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric

import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from data_loader_2d import get_dataloaders
import wandb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
batch_size = 4 if not args.debug else 2
train_loader, val_loader = get_dataloaders(
    data_dir="./data/raw", batch_size=batch_size, debug=args.debug
)

# ==============================
# Model
# ==============================
model = deeplabv3_resnet50(weights=None, num_classes=2)
# 单通道输入（CT）
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
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)  # 训练时：预测 [B,2,H,W], 标签 [B,1,H,W]
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# ==============================
# Metrics
# ==============================
dice_metric        = DiceMetric(include_background=False, reduction="none")
precision_metric   = ConfusionMatrixMetric(metric_name="precision",   reduction="mean", include_background=False)
recall_metric      = ConfusionMatrixMetric(metric_name="recall",      reduction="mean", include_background=False)
specificity_metric = ConfusionMatrixMetric(metric_name="specificity", reduction="mean", include_background=False)

# ==============================
# AMP Scaler
# ==============================
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ==============================
# Training Loop
# ==============================
for epoch in trange(start_epoch, num_epochs, desc="Total Progress"):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].to(device)              # [B,1,H,W]
        masks  = batch["label"].unsqueeze(1).to(device).long()  # [B,1,H,W]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)["out"]   # [B,2,H,W]
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
        empty_gt_skipped = 0
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)                 # [B,1,H,W]
            masks  = batch["label"].unsqueeze(1).to(device)    # [B,1,H,W] (索引在通道维)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)["out"]                  # [B,2,H,W]
                loss = loss_fn(logits, masks)                  # DiceCE 仍然用 one-hot 计算
            val_loss += loss.item()

            # ===== 统一的“索引图”指标计算 =====
            pred_idx = logits.argmax(dim=1)                    # [B,H,W] {0,1}
            true_idx = masks.squeeze(1)                        # [B,H,W] {0,1}

            for p, t in zip(pred_idx, true_idx):
                # 跳过全空 GT（否则 Dice/IoU 会假性偏高/偏低）
                if t.sum() == 0:
                    empty_gt_skipped += 1
                    continue

                tp = ((p == 1) & (t == 1)).sum().float()
                fp = ((p == 1) & (t == 0)).sum().float()
                fn = ((p == 0) & (t == 1)).sum().float()

                denom_dice = (2 * tp + fp + fn)
                denom_iou  = (tp + fp + fn)

                dice = (2 * tp) / (denom_dice + 1e-6)
                iou  = tp / (denom_iou + 1e-6)

                # 记录
                ious.append(iou.item())
                # 你也可以顺便求平均 Dice（这里直接重用 dice_metric 也行）
                dice_metric(y_pred=[(p == 1).unsqueeze(0).unsqueeze(0).float()],
                            y=[t.eq(1).unsqueeze(0).unsqueeze(0).float()])

    avg_val_loss = val_loss / max(1, len(val_loader))
    fg_dice = float(torch.as_tensor(dice_metric.aggregate()).mean().item()) if len(ious) > 0 else float('nan')
    miou_val = float(torch.tensor(ious).mean().item()) if ious else float('nan')

    print(f"Val Loss: {avg_val_loss:.4f}, Dice={fg_dice:.4f}, mIoU={miou_val:.4f}")
    wandb.log({
        "Loss/val": avg_val_loss,
        "Dice/val": fg_dice,
        "mIoU/val": miou_val,
        "Val/empty_gt_skipped": empty_gt_skipped,
    }, step=epoch+1)

    if fg_dice > best_dice:
        best_dice = fg_dice
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    scheduler.step()

wandb.finish()
writer.close()
