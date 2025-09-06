import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    EnsureTyped, AsDiscreted, SaveImaged
)
from monai.data import Dataset, DataLoader, list_data_collate

def build_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
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
    if label_path is not None and os.path.isfile(label_path):
        data["label"] = label_path
        keys.append("label")

    transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        # （与训练一致的强度归一化）
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys),
    ])
    ds = Dataset([data], transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    return loader

def save_nifti(pred_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saver = SaveImaged(
        keys=["pred"],
        meta_keys=["image_meta_dict"],
        output_dir=out_dir,
        output_postfix="pred",
        resample=False,  # 保持与原图同空间
        separate_folder=False
    )
    saver(pred_dict)
    return pred_dict["pred_meta_dict"]["filename_or_obj"]

def visualize_mid_slices(image_np, pred_np, gt_np, out_png):
    """
    image_np: [1, D, H, W]  (after EnsureChannelFirstd)
    pred_np:  [2, D, H, W]  (one-hot or logits argmax→one-hot)
    gt_np:    [2, D, H, W] or None
    """
    img = image_np[0]       # [D, H, W]
    pred_fg = pred_np[1]    # [D, H, W]

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
    plt.imshow(pred_slice, alpha=0.4)  # 叠加预测
    plt.title("Pred (overlay, mid)"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="3D UNet Inference & Visualization")
    parser.add_argument("--model", required=True, help="Path to UNet .pth (best_model.pth or latest_model.pth)")
    parser.add_argument("--image", required=True, help="Path to image .nii.gz")
    parser.add_argument("--label", default=None, help="(Optional) Path to label .nii.gz for visualization")
    parser.add_argument("--out_dir", default="./pred_out", help="Output directory")
    parser.add_argument("--roi", default="128,128,64", help="Sliding window ROI size, e.g., 128,128,64")
    parser.add_argument("--overlap", type=float, default=0.25, help="SWI overlap")
    parser.add_argument("--sw_batch_size", type=int, default=1, help="SWI batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inference on {device}")

    # Model
    model = build_model(device)
    load_weights(model, args.model, device)
    model.eval()

    # Data
    loader = get_infer_loader(args.image, args.label)

    # ROI
    roi_size = tuple(int(x) for x in args.roi.split(","))

    # Post transforms（argmax 到 one-hot，便于可视化/保存）
    post_proc = Compose([
        AsDiscreted(keys=["pred"], argmax=True, to_onehot=2),
    ])

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)                # [1, 1, D, H, W]
            meta = batch["image_meta_dict"]

            logits = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap
            )                                                  # [1, 2, D, H, W]

            pred_dict = {
                "image": batch["image"],              # for meta & saving reference
                "image_meta_dict": meta,
                "pred": logits.cpu(),                 # keep on CPU for post + save
            }

            pred_dict = post_proc(pred_dict)          # argmax→one-hot: [1,2,D,H,W]

            # Save NIfTI
            os.makedirs(args.out_dir, exist_ok=True)
            saved_path = save_nifti(pred_dict, args.out_dir)
            print(f"[INFO] Saved prediction NIfTI to: {saved_path}")

            # Visualization PNG
            img_np  = pred_dict["image"].numpy()[0]        # [1,D,H,W]
            pred_np = pred_dict["pred"].numpy()[0]         # [2,D,H,W]
            gt_np   = None
            if args.label and "label" in batch:
                # 将 label 也转为 one-hot 以统一可视化
                from monai.transforms import AsDiscrete
                gt_onehot = AsDiscrete(to_onehot=2)(batch["label"])
                gt_np = gt_onehot.numpy()[0]               # [2,D,H,W]

            base = os.path.splitext(os.path.basename(args.image))[0].replace(".nii", "")
            out_png = os.path.join(args.out_dir, f"{base}_viz.png")
            visualize_mid_slices(img_np, pred_np, gt_np, out_png)
            print(f"[INFO] Saved visualization to: {out_png}")

if __name__ == "__main__":
    main()
