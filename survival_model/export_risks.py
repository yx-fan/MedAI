#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export survival model risk scores for train/val sets.

Output:
  - runs/survival/train_risks.csv
  - runs/survival/val_risks.csv
"""

import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import SurvivalDataset
from models import ImageEncoder2_5D, ClinicalMLP, MultiModalCox


@torch.no_grad()
def export_risks(loader, img_encoder, clin_mlp, fusion, device, out_path):
    risks, times, events, pids = [], [], [], []
    for batch in loader:
        if len(batch) == 5:
            images, clinical, time, event, pid = batch
        else:
            images, clinical, time, event = batch
            pid = [None] * len(images)

        images = images.to(device)
        clinical = clinical.to(device)

        z_img = img_encoder(images)
        z_clin = clin_mlp(clinical)
        risk = fusion(z_img, z_clin)

        risks.extend(risk.cpu().numpy().flatten().tolist())
        times.extend(time.numpy().flatten().tolist())
        events.extend(event.numpy().flatten().tolist())
        pids.extend([str(x) for x in pid])

    df = pd.DataFrame({
        "patient_id": pids,
        "risk": risks,
        "time": times,
        "event": events
    })
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved risks to {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_ds = SurvivalDataset(
        meta_csv=args.meta_csv,
        clinical_csv=args.clinical_csv,
        processed_dir=args.processed_dir,
        split="train",
        agg="mean"
    )
    val_ds = SurvivalDataset(
        meta_csv=args.meta_csv,
        clinical_csv=args.clinical_csv,
        processed_dir=args.processed_dir,
        split="val",
        agg="mean"
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Build models
    sample = train_ds[0]
    img_slices, clin_features = sample[0], sample[1]
    N = img_slices.shape[1]
    clin_dim = clin_features.numel()

    img_encoder = ImageEncoder2_5D(in_slices=N, out_dim=args.img_embed_dim).to(device)
    clin_mlp = ClinicalMLP(in_dim=clin_dim, hidden=128, out_dim=args.clin_embed_dim, dropout=0.1).to(device)
    fusion = MultiModalCox(img_embed_dim=args.img_embed_dim, clin_embed_dim=args.clin_embed_dim, hidden=128).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    img_encoder.load_state_dict(ckpt["img_encoder"])
    clin_mlp.load_state_dict(ckpt["clin_mlp"])
    fusion.load_state_dict(ckpt["fusion"])
    print(f"[INFO] Loaded checkpoint from {args.ckpt_path} (epoch={ckpt['epoch']}, val_c={ckpt['val_c']:.4f})")

    img_encoder.eval(); clin_mlp.eval(); fusion.eval()

    # Export risks
    os.makedirs(args.out_dir, exist_ok=True)
    export_risks(train_loader, img_encoder, clin_mlp, fusion, device, os.path.join(args.out_dir, "train_risks.csv"))
    export_risks(val_loader, img_encoder, clin_mlp, fusion, device, os.path.join(args.out_dir, "val_risks.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export risk scores from best Cox model")
    parser.add_argument("--ckpt_path", type=str, default="runs/survival/best_mm_cox.pth",
                        help="Path to best Cox checkpoint")
    parser.add_argument("--meta_csv", type=str, default="data/processed/train_2_5d/meta.csv")
    parser.add_argument("--clinical_csv", type=str, default="data/processed/clinical_processed.csv")
    parser.add_argument("--processed_dir", type=str, default="data/processed/train_2_5d/")
    parser.add_argument("--out_dir", type=str, default="runs/survival")
    parser.add_argument("--img_embed_dim", type=int, default=256)
    parser.add_argument("--clin_embed_dim", type=int, default=128)
    args = parser.parse_args()
    main(args)
