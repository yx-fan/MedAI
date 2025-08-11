import os
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess_utils import (
    load_nifti,
    get_processed_2d,
    get_processed_2_5d,
    get_processed_3d_patch,
    DEFAULT_MARGIN
)


def generate_dataset(images_dir,
                     masks_dir=None,
                     output_dir="data/processed",
                     mode="train",
                     format="2.5d",
                     N=5,
                     margin=DEFAULT_MARGIN,
                     out_size=(256, 256),
                     out_size_3d=(128, 128, 64),
                     background_ratio=0.1,
                     augment=False):
    """
    Generate processed dataset for training or inference.

    Args:
        images_dir: folder containing CT images (.nii.gz)
        masks_dir: folder containing masks (.nii.gz) (required for mode='train')
        output_dir: output folder to save processed npy files
        mode: 'train' (needs mask) | 'predict' (no mask)
        format: '2d' | '2.5d' | '3d'
        N: number of slices for 2.5D stack (odd number)
        margin: pixels around ROI
        out_size: output size (H, W) for 2D / 2.5D
        out_size_3d: output size (H, W, D) for 3D patches
        background_ratio: proportion of background slices to keep in training mode
        augment: apply data augmentation (only when mode='train')
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir) if masks_dir else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_info = []

    for ct_path in images_dir.glob("*.nii.gz"):
        pid = ct_path.name.replace(".nii.gz", "")
        mask_path = masks_dir / f"{pid}.nii.gz" if masks_dir else None

        # Load CT and mask
        ct, _, _ = load_nifti(ct_path)
        mask = None
        if mask_path and mask_path.exists():
            mask, _, _ = load_nifti(mask_path)

        if mode == "train" and mask is None:
            print(f"[WARN] Missing mask for {pid}, skipping.")
            continue

        # Select slice indices
        if mask is not None:
            tumor_slices = np.where(np.any(mask > 0, axis=(0, 1)))[0]
            background_slices = np.setdiff1d(np.arange(ct.shape[2]), tumor_slices)

            if mode == "train" and background_ratio > 0:
                num_bg_keep = int(len(background_slices) * background_ratio)
                bg_keep_slices = np.random.choice(background_slices, size=num_bg_keep, replace=False)
                slice_indices = np.sort(np.concatenate([tumor_slices, bg_keep_slices]))
            else:
                slice_indices = np.arange(ct.shape[2])
        else:
            # Predict mode → all slices
            slice_indices = np.arange(ct.shape[2])

        for idx in slice_indices:
            if format == "2d":
                processed = get_processed_2d(pid, ct, mask, idx, margin, out_size, mode=mode, augment=augment)
            elif format == "2.5d":
                processed = get_processed_2_5d(pid, ct, mask, idx, N, margin, out_size, mode=mode, augment=augment)
            elif format == "3d":
                processed = get_processed_3d_patch(pid, ct, mask, margin, out_size_3d, mode=mode)
            else:
                raise ValueError("Invalid format")

            if processed is None:
                continue

            npy_name = (
                f"{pid}_slice{idx}.npy" if format != "3d" else f"{pid}_3d.npy"
            )
            np.save(output_dir / npy_name, processed)

            has_tumor = (
                int(mask is not None and np.any(mask[:, :, idx] > 0))
                if format != "3d"
                else None
            )

            meta_info.append({
                "patient_id": pid,
                "slice_idx": idx if format != "3d" else None,
                "ct_file": str(ct_path),
                "mask_file": str(mask_path) if mask_path else None,
                "npy_file": npy_name,
                "format": format,
                "has_tumor": has_tumor
            })

    # Save meta.csv
    pd.DataFrame(meta_info).to_csv(output_dir / "meta.csv", index=False)
    print(f"✅ Done! Processed {len(meta_info)} samples.")
