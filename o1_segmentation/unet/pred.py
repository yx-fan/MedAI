import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, EnsureTyped
)
from monai.data import Dataset, DataLoader, list_data_collate


def build_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
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


def get_infer_loader(image_path, label_path=None):
    data = {"image": image_path}
    keys = ["image"]
    if label_path and os.path.isfile(label_path):
        data["label"] = label_path
        keys.append("label")

    transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys),
    ])
    ds = Dataset([data], transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    return loader


def save_nifti_simple(pred_tensor, out_path):
    if pred_tensor.ndim == 5:
        pred_tensor = pred_tensor.squeeze(0).squeeze(0)
    elif pred_tensor.ndim == 4:
        pred_tensor = pred_tensor.squeeze(0)
    else:
        raise ValueError(f"Unexpected shape: {pred_tensor.shape}")

    pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(pred_np, affine), out_path)
    print(f"[INFO] Saved NIfTI to: {out_path}")


def visualize_mid_slices(image_np, pred_np, gt_np, out_png):
    img = image_np[0]
    pred_fg = pred_np[1]
    mid = img.shape[0] // 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3 if gt_np is not None else 2, 1)
    plt.imshow(img[mid], cmap="gray")
    plt.title("Image"); plt.axis("off")

    if gt_np is not None:
        gt_fg = gt_np[1]
        plt.subplot(1, 3, 2)
        plt.imshow(gt_fg[mid], cmap="gray")
        plt.title("GT"); plt.axis("off")
        idx = 3
    else:
        idx = 2

    plt.subplot(1, 3 if gt_np is not None else 2, idx)
    plt.imshow(img[mid], cmap="gray")
    plt.imshow(pred_fg[mid], alpha=0.4)
    plt.title("Pred Overlay"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Inference for 3D UNet")
    parser.add_argument("--model", required=True, help="Path to model .pth")
    parser.add_argument("--image_dir", required=True, help="Directory with images (.nii.gz)")
    parser.add_argument("--label_dir", default=None, help="Optional label directory for visualization")
    parser.add_argument("--out_dir", default="./pred_out", help="Output directory")
    parser.add_argument("--roi", default="128,128,64", help="Sliding window ROI")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap")
    parser.add_argument("--sw_batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running inference on {device}")

    model = build_model(device)
    load_weights(model, args.model, device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    roi_size = tuple(int(x) for x in args.roi.split(","))

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(".nii.gz")])
    print(f"[INFO] Found {len(image_files)} images in {args.image_dir}")

    for img_file in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(args.image_dir, img_file)
        label_path = None
        if args.label_dir:
            label_path = os.path.join(args.label_dir, img_file)

        loader = get_infer_loader(img_path, label_path)
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                logits = sliding_window_inference(
                    images,
                    roi_size=roi_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.overlap
                )
                pred = torch.argmax(logits, dim=1, keepdim=True).cpu()

                base = os.path.splitext(os.path.basename(img_file))[0].replace(".nii", "")
                out_nii = os.path.join(args.out_dir, f"{base}_pred.nii.gz")
                save_nifti_simple(pred, out_nii)

                img_np = batch["image"].numpy()[0]
                pred_bin = pred.numpy()[0, 0]
                pred_np = np.stack([1 - pred_bin, pred_bin], axis=0)

                gt_np = None
                if "label" in batch:
                    lbl = batch["label"].numpy()[0, 0]
                    gt_np = np.stack([1 - lbl, lbl], axis=0)

                out_png = os.path.join(args.out_dir, f"{base}_viz.png")
                visualize_mid_slices(img_np, pred_np, gt_np, out_png)

    print(f"[INFO] All predictions saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
