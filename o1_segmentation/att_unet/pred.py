import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    EnsureTyped
)
from monai.data import Dataset, DataLoader, list_data_collate


# =============================
# Model
# =============================
def build_model(device):
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    return model


def load_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded checkpoint dict['model'] from {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"[INFO] Loaded state_dict from {ckpt_path}")


# =============================
# Data Loader
# =============================
def get_infer_loader(image_dir, label_dir=None):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
    data = []
    for img_path in image_files:
        case_id = os.path.splitext(os.path.basename(img_path))[0]
        item = {"image": img_path, "case_id": case_id}
        if label_dir is not None:
            lbl_path = os.path.join(label_dir, os.path.basename(img_path))
            if os.path.isfile(lbl_path):
                item["label"] = lbl_path
        data.append(item)

    keys = ["image", "label"] if label_dir else ["image"]
    transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys),
    ])
    ds = Dataset(data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    return loader


# =============================
# Save / Visualization
# =============================
def save_nifti_simple(pred_tensor, out_path):
    if pred_tensor.ndim == 5:  # [1,1,D,H,W]
        pred_tensor = pred_tensor.squeeze(0).squeeze(0)
    elif pred_tensor.ndim == 4:  # [1,D,H,W]
        pred_tensor = pred_tensor.squeeze(0)
    else:
        raise ValueError(f"Unexpected shape: {pred_tensor.shape}")

    pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(pred_np, affine), out_path)
    print(f"[INFO] Saved prediction NIfTI to: {out_path}, shape={pred_np.shape}")


def visualize_mid_slices(image_np, pred_np, gt_np, out_png):
    img = image_np[0]       # [D,H,W]
    pred_fg = pred_np[1]    # 前景通道

    mid = img.shape[0] // 2
    img_slice = img[mid]
    pred_slice = pred_fg[mid]

    ncols = 3 if gt_np is not None else 2
    plt.figure(figsize=(4 * ncols, 4))

    plt.subplot(1, ncols, 1)
    plt.imshow(img_slice, cmap="gray")
    plt.title("Image (mid)"); plt.axis("off")

    if gt_np is not None:
        gt_fg = gt_np[1]
        plt.subplot(1, ncols, 2)
        plt.imshow(gt_fg[mid], cmap="gray")
        plt.title("GT (mid)"); plt.axis("off")
        ax_idx = 3
    else:
        ax_idx = 2

    plt.subplot(1, ncols, ax_idx)
    plt.imshow(img_slice, cmap="gray")
    plt.imshow(pred_slice, alpha=0.4)
    plt.title("Pred (overlay, mid)"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =============================
# Dice Calculation (per case)
# =============================
def compute_dice(pred_np, gt_np):
    intersection = np.logical_and(pred_np, gt_np).sum()
    dice = (2. * intersection) / (pred_np.sum() + gt_np.sum() + 1e-8)
    return dice


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(description="3D Attention UNet Batch Inference & Per-Case Dice")
    parser.add_argument("--model", required=True, help="Path to Attention UNet .pth checkpoint")
    parser.add_argument("--image_dir", required=True, help="Directory of images (.nii.gz)")
    parser.add_argument("--label_dir", default=None, help="Directory of labels (.nii.gz)")
    parser.add_argument("--out_dir", default="./pred_out", help="Output directory")
    parser.add_argument("--roi", default="128,128,64", help="Sliding window ROI size, e.g., 128,128,64")
    parser.add_argument("--overlap", type=float, default=0.25, help="SWI overlap")
    parser.add_argument("--sw_batch_size", type=int, default=1, help="SWI batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inference on {device}")

    model = build_model(device)
    load_weights(model, args.model, device)
    model.eval()

    loader = get_infer_loader(args.image_dir, args.label_dir)
    roi_size = tuple(int(x) for x in args.roi.split(","))

    dice_scores = []
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            case_id = batch["case_id"][0] if "case_id" in batch else "unknown"

            logits = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap
            )

            pred = torch.argmax(logits, dim=1, keepdim=True).cpu()

            out_nii = os.path.join(args.out_dir, f"{case_id}_pred.nii.gz")
            save_nifti_simple(pred, out_nii)

            # numpy 用于可视化和 Dice
            img_np = batch["image"].numpy()[0]
            pred_bin = pred.numpy()[0, 0]
            pred_np = np.stack([1 - pred_bin, pred_bin], axis=0).astype(np.uint8)

            gt_np = None
            if "label" in batch:
                lbl = batch["label"].numpy()[0, 0]
                gt_np = np.stack([1 - lbl, lbl], axis=0).astype(np.uint8)

                # 计算 Dice
                dice = compute_dice(pred_bin, lbl)
                dice_scores.append(dice)
                print(f"[RESULT] Dice for {case_id}: {dice:.4f}")

            out_png = os.path.join(args.out_dir, f"{case_id}_viz.png")
            visualize_mid_slices(img_np, pred_np, gt_np, out_png)
            print(f"[INFO] Saved visualization to: {out_png}")

    if args.label_dir and dice_scores:
        mean_dice = np.mean(dice_scores)
        print(f"[RESULT] Mean Dice over dataset: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
