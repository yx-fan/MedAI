#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from dataset import MultiModalDataset
from models import ImageEncoder2_5D, ClinicalMLP, MultiModalNet, cox_ph_loss

# -----------------------------
# C-index (Harrell's C)
# -----------------------------
def c_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    r = risk.detach().cpu().numpy()
    t = time.detach().cpu().numpy()
    e = event.detach().cpu().numpy().astype(bool)

    n_conc, n_tied, n_total = 0, 0, 0
    for i in range(len(r)):
        for j in range(len(r)):
            if t[i] == t[j]:
                continue
            if t[i] < t[j] and e[i]:
                n_total += 1
                if r[i] > r[j]:
                    n_conc += 1
                elif r[i] == r[j]:
                    n_tied += 1
            elif t[j] < t[i] and e[j]:
                n_total += 1
                if r[j] > r[i]:
                    n_conc += 1
                elif r[i] == r[j]:
                    n_tied += 1
    if n_total == 0:
        return float("nan")
    return (n_conc + 0.5 * n_tied) / n_total


# -----------------------------
# Train / Val loop
# -----------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # -----------------------------
    # Load datasets
    # -----------------------------
    full_train_ds = MultiModalDataset(
        meta_csv=args.meta_csv,
        clinical_csv=args.clinical_csv,
        processed_dir=args.processed_dir,
        split="train",
        agg="mean"
    )
    val_ds = MultiModalDataset(
        meta_csv=args.meta_csv,
        clinical_csv=args.clinical_csv,
        processed_dir=args.processed_dir,
        split="val",
        agg="mean"
    )

    if len(val_ds) == 0:
        indices = list(range(len(full_train_ds)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_ds = Subset(full_train_ds, train_idx)
        val_ds   = Subset(full_train_ds, val_idx)
        print(f"[INFO] Auto-split train into {len(train_ds)} train / {len(val_ds)} val")
    else:
        train_ds = full_train_ds

    if args.debug:
        train_ds = Subset(train_ds, list(range(min(50, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(20, len(val_ds)))))
        args.epochs = 3
        args.batch_size = 2
        print(f"[DEBUG] Using subset: {len(train_ds)} train, {len(val_ds)} val, {args.epochs} epochs")

    # -----------------------------
    # Dataloaders
    # -----------------------------
    sample = train_ds[0]
    sample_img, sample_clin = sample[0], sample[1]
    N = sample_img.shape[1]
    clin_dim = sample_clin.numel()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # -----------------------------
    # Model
    # -----------------------------
    img_encoder = ImageEncoder2_5D(in_slices=N, out_dim=args.img_embed_dim).to(device)
    clin_mlp = ClinicalMLP(in_dim=clin_dim, hidden=128, out_dim=args.clin_embed_dim).to(device)
    model = MultiModalNet(args.img_embed_dim, args.clin_embed_dim, hidden=128, num_classes=2).to(device)

    params = list(img_encoder.parameters()) + list(clin_mlp.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_metric = -1.0
    patience_counter = 0
    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] ===== Epoch {epoch}/{args.epochs} =====")
        img_encoder.train(); clin_mlp.train(); model.train()
        tr_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
        for step, batch in enumerate(pbar):
            if len(batch) == 6:  # multimodal dataset: (img, clin, time, event, ln_label, pid)
                images, clinical, time, event, ln_label, pid = batch
            else:
                raise ValueError("Dataset must return 6 elements for multitask training")

            images, clinical = images.to(device), clinical.to(device)
            time, event, ln_label = time.to(device).float(), event.to(device).float(), ln_label.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                z_img = img_encoder(images)
                z_clin = clin_mlp(clinical)

                if args.task == "survival":
                    risk = model(z_img, z_clin, task="survival")
                    loss = cox_ph_loss(risk, time, event)

                elif args.task == "ln_classification":
                    logits = model(z_img, z_clin, task="ln_classification")
                    loss = F.cross_entropy(logits, ln_label)

                else:
                    raise ValueError(f"Unknown task: {args.task}")

            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            tr_loss += loss.item()

            if step % 10 == 0:
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        tr_loss /= max(1, len(train_loader))
        print(f"[INFO] Train Loss={tr_loss:.4f}")

        # -----------------------------
        # Validation
        # -----------------------------
        img_encoder.eval(); clin_mlp.eval(); model.eval()
        val_losses, preds, targets, probs = [], [], [], []
        risks, times, events = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False):
                images, clinical, time, event, ln_label, pid = batch
                images, clinical = images.to(device), clinical.to(device)
                time, event, ln_label = time.to(device).float(), event.to(device).float(), ln_label.to(device).long()

                z_img = img_encoder(images)
                z_clin = clin_mlp(clinical)

                if args.task == "survival":
                    risk = model(z_img, z_clin, task="survival")
                    loss = cox_ph_loss(risk, time, event)
                    val_losses.append(loss.item())
                    risks.append(risk.cpu()); times.append(time.cpu()); events.append(event.cpu())

                elif args.task == "ln_classification":
                    logits = model(z_img, z_clin, task="ln_classification")
                    loss = F.cross_entropy(logits, ln_label)
                    val_losses.append(loss.item())
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    targets.extend(ln_label.cpu().numpy())
                    probs.extend(F.softmax(logits, dim=1)[:,1].cpu().numpy())  # 🔥 概率分数

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        if args.task == "survival":
            risks = torch.cat(risks) if risks else torch.tensor([])
            times = torch.cat(times) if times else torch.tensor([])
            events = torch.cat(events) if events else torch.tensor([])
            val_c = c_index(risks, times, events) if len(risks) else float("nan")
            metric = val_c
            print(f"[Epoch {epoch:03d}] TrainLoss={tr_loss:.4f} | ValLoss={val_loss:.4f} | Val C-index={val_c:.4f}")

        elif args.task == "ln_classification":
            acc = (np.array(preds) == np.array(targets)).mean() if preds else 0.0
            auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else float("nan")
            metric = auc  # 🔥 用 AUC 做 early stop
            print(f"[Epoch {epoch:03d}] TrainLoss={tr_loss:.4f} | "
                  f"ValLoss={val_loss:.4f} | Val Acc={acc:.4f} | Val AUC={auc:.4f}")

            if len(set(targets)) > 1:
                fpr, tpr, _ = roc_curve(targets, probs)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0,1], [0,1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve (Epoch {epoch})")
                plt.legend(loc="lower right")
                plt.tight_layout()
                roc_path = os.path.join(args.out_dir, f"roc_epoch{epoch:03d}.png")
                plt.savefig(roc_path, dpi=300)
                plt.close()
                print(f"[INFO] ROC curve saved to {roc_path}")

        scheduler.step()

        # save best + early stopping
        if not np.isnan(metric) and metric > best_val_metric:
            best_val_metric = metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "img_encoder": img_encoder.state_dict(),
                "clin_mlp": clin_mlp.state_dict(),
                "model": model.state_dict(),
                "val_metric": metric,
                "task": args.task
            }, os.path.join(args.out_dir, f"best_multitask_{args.task}.pth"))
            print(f"[INFO] Saved best model ({args.task}, metric={metric:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"[EARLY STOP] No improvement for {args.patience} epochs. Stopping training.")
            break

    print(f"[DONE] Best Val metric ({args.task}): {best_val_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_csv", default="data/processed/train_2_5d/meta.csv", type=str)
    parser.add_argument("--clinical_csv", default="data/processed/clinical_processed_multimodal.csv", type=str)
    parser.add_argument("--processed_dir", default="data/processed/train_2_5d/", type=str)
    parser.add_argument("--out_dir", default="runs/multitask", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--img_embed_dim", default=256, type=int)
    parser.add_argument("--clin_embed_dim", default=128, type=int)
    parser.add_argument("--task", choices=["survival", "ln_classification"], required=True,
                        help="Which task to train: survival or ln_classification")
    parser.add_argument("--patience", default=10, type=int,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer samples/epochs")
    args = parser.parse_args()
    run(args)


#sample usage:
# python multiModal/train_multimodal.py \
#     --meta_csv data/processed/train_2_5d/meta.csv \
#     --clinical_csv data/processed/clinical_processed_multimodal.csv \
#     --processed_dir data/processed/train_2_5d/ \
#     --out_dir runs/ln_classification \
#     --epochs 50 \
#     --batch_size 8 \
#     --lr 1e-3 \
#     --img_embed_dim 256 \
#     --clin_embed_dim 128 \
#     --task ln_classification


# Example usage:
# python survival_model/train_multitask.py \
#     --meta_csv data/processed/train_2_5d/meta.csv \
#     --clinical_csv data/processed/clinical_processed_multimodal.csv \
#     --processed_dir data/processed/train_2_5d/ \
#     --out_dir runs/survival \
#     --epochs 50 \
#     --batch_size 8 \
#     --lr 1e-3 \
#     --img_embed_dim 256 \
#     --clin_embed_dim 128 \
#     --task survival
