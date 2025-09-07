import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 你之前的 Dataset（确保能 import 到）
from dataset import SurvivalDataset
from models import ImageEncoder2_5D, ClinicalMLP, MultiModalCox, cox_ph_loss

# -----------------------------
# C-index (Harrell's C)
# -----------------------------
def c_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """
    risk: higher means higher risk (shorter survival)
    Only comparable pairs where min(time_i, time_j) has event=1 contribute.
    """
    r = risk.detach().cpu().numpy()
    t = time.detach().cpu().numpy()
    e = event.detach().cpu().numpy().astype(bool)

    n_conc, n_tied, n_total = 0, 0, 0
    # O(N^2) baseline; for moderate N fine. For large N consider optimized versions.
    for i in range(len(r)):
        for j in range(len(r)):
            if t[i] == t[j]:
                continue
            # pair is comparable if the shorter time had an event
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

    # Datasets
    train_ds = SurvivalDataset(
        meta_csv=args.meta_csv,
        clinical_csv=args.clinical_csv,
        processed_dir=args.processed_dir,
        split="train",
        agg="mean"   # you can switch to "max"
    )
    val_ds = SurvivalDataset(
        meta_csv=args.meta_csv,
        clinical_csv=args.clinical_csv,
        processed_dir=args.processed_dir,
        split="val",
        agg="mean"
    )

    # Infer N for image encoder from one sample
    sample_img, sample_clin, _, _ = train_ds[0]
    # sample_img: (2, N, H, W)
    N = sample_img.shape[1]
    clin_dim = sample_clin.numel()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Models
    img_encoder = ImageEncoder2_5D(in_slices=N, out_dim=args.img_embed_dim).to(device)
    clin_mlp = ClinicalMLP(in_dim=clin_dim, hidden=128, out_dim=args.clin_embed_dim, dropout=0.1).to(device)
    fusion = MultiModalCox(img_embed_dim=args.img_embed_dim, clin_embed_dim=args.clin_embed_dim, hidden=128).to(device)

    params = list(img_encoder.parameters()) + list(clin_mlp.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_c = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        img_encoder.train(); clin_mlp.train(); fusion.train()
        tr_loss = 0.0
        for images, clinical, time, event in train_loader:
            images = images.to(device)           # (B, 2, N, H, W)
            clinical = clinical.to(device)       # (B, F)
            time = time.to(device).float()
            event = event.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z_img = img_encoder(images)
                z_clin = clin_mlp(clinical)
                risk = fusion(z_img, z_clin)
                loss = cox_ph_loss(risk, time, event)

            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))

        # Validation
        img_encoder.eval(); clin_mlp.eval(); fusion.eval()
        val_losses, risks, times, events = [], [], [], []
        with torch.no_grad():
            for images, clinical, time, event in val_loader:
                images = images.to(device)
                clinical = clinical.to(device)
                time = time.to(device).float()
                event = event.to(device).float()

                z_img = img_encoder(images)
                z_clin = clin_mlp(clinical)
                risk = fusion(z_img, z_clin)
                loss = cox_ph_loss(risk, time, event)

                val_losses.append(loss.item())
                risks.append(risk.detach().cpu())
                times.append(time.detach().cpu())
                events.append(event.detach().cpu())

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        risks = torch.cat(risks) if risks else torch.tensor([])
        times = torch.cat(times) if times else torch.tensor([])
        events = torch.cat(events) if events else torch.tensor([])
        val_c = c_index(risks, times, events) if len(risks) else float("nan")

        scheduler.step()

        print(f"[Epoch {epoch:03d}] TrainLoss={tr_loss:.4f} | ValLoss={val_loss:.4f} | Val C-index={val_c:.4f}")

        # Save best
        if not np.isnan(val_c) and val_c > best_val_c:
            best_val_c = val_c
            torch.save({
                "epoch": epoch,
                "img_encoder": img_encoder.state_dict(),
                "clin_mlp": clin_mlp.state_dict(),
                "fusion": fusion.state_dict(),
                "val_c": val_c
            }, os.path.join(args.out_dir, "best_mm_cox.pth"))
            print(f"[INFO] Saved best model (C-index={val_c:.4f})")

    print(f"[DONE] Best Val C-index: {best_val_c:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_csv", default="data/processed/meta.csv", type=str)
    parser.add_argument("--clinical_csv", default="data/processed/clinical_processed.csv", type=str)
    parser.add_argument("--processed_dir", default="data/processed", type=str)
    parser.add_argument("--out_dir", default="runs/survival", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--img_embed_dim", default=256, type=int)
    parser.add_argument("--clin_embed_dim", default=128, type=int)
    args = parser.parse_args()
    run(args)
