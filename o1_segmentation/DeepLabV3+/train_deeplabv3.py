import os
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models.segmentation import deeplabv3_resnet50

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

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
num_epochs = base_epochs + max(0, args.extra_epochs)
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
    config={"epochs": num_epochs, "batch_size": 4 if not args.debug else 2, "lr": learning_rate, "arch": "DeepLabV3"},
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
dice_metric = DiceMetric(include_background=False, reduction="none")

# ==============================
# AMP Scaler
# ==============================
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ==============================
# (Optional) Resume
# ==============================
if args.resume and os.path.isfile(args.resume):
    state = torch.load(args.resume, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Resumed from {args.resume}")

# ==============================
# Training Loop
# ==============================
for epoch in trange(start_epoch, num_epochs, desc="Total Progress"):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch["image"].to(device)                    # [B,1,H,W]
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
    dice_metric.reset()
    ious = []

    # 手动累计 TP/FP/FN/TN（前景=类1）
    tp_total = fp_total = fn_total = tn_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)                       # [B,1,H,W]
            masks  = batch["label"].unsqueeze(1).to(device).long()   # [B,1,H,W]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)["out"]   # [B,2,H,W]
                loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            # —— 显式把预测与标签都转为 one-hot 的 [B,2,H,W] —— #
            pred = torch.argmax(outputs, dim=1)  # [B,H,W]
            y_pred_oh = torch.nn.functional.one_hot(pred, num_classes=2).permute(0, 3, 1, 2).float()  # [B,2,H,W]

            lbl  = masks.squeeze(1)  # [B,H,W]
            y_oh = torch.nn.functional.one_hot(lbl,  num_classes=2).permute(0, 3, 1, 2).float()       # [B,2,H,W]

            # 拆成 list 以适配 MONAI 度量器（每个样本形状 [1,2,H,W]）
            y_pred_list = [p.unsqueeze(0) for p in torch.unbind(y_pred_oh, dim=0)]
            y_list      = [t.unsqueeze(0) for t in torch.unbind(y_oh,      dim=0)]

            # Dice
            dice_metric(y_pred=y_pred_list, y=y_list)

            # 手动算 P / R / Spe 与 mIoU（基于布尔张量）
            for p, t in zip(y_pred_list, y_list):
                # p,t: [1,2,H,W]，取前景通道 -> [H,W] -> 展平
                pb = (p[:, 1] > 0.5).flatten()   # bool
                tb = (t[:, 1] > 0.5).flatten()   # bool

                tp = (pb & tb).sum().item()
                fp = (pb & ~tb).sum().item()
                fn = (~pb & tb).sum().item()
                tn = (~pb & ~tb).sum().item()

                tp_total += tp; fp_total += fp; fn_total += fn; tn_total += tn

                inter = (pb & tb).sum().item()
                union = (pb | tb).sum().item()
                if union > 0:
                    ious.append(inter / union)

    avg_val_loss = val_loss / max(1, len(val_loader))
    fg_dice = float(torch.as_tensor(dice_metric.aggregate()).mean().item())

    precision_val   = tp_total / (tp_total + fp_total + 1e-8)
    recall_val      = tp_total / (tp_total + fn_total + 1e-8)
    specificity_val = tn_total / (tn_total + fp_total + 1e-8)
    miou_val        = float(sum(ious) / len(ious)) if ious else 0.0

    # 额外 sanity check：用 IoU 反推 Dice（不参与日志，可自行打印核对）
    # dice_from_iou = (2 * miou_val) / (1 + miou_val + 1e-8)

    print(f"Val Loss: {avg_val_loss:.4f}, Dice={fg_dice:.4f}, "
          f"Precision={precision_val:.4f}, Recall={recall_val:.4f}, "
          f"Specificity={specificity_val:.4f}, mIoU={miou_val:.4f}")

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
