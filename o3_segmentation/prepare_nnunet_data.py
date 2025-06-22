import os
from pathlib import Path
import shutil
import nibabel as nib
import numpy as np

RAW_IMAGE_DIR = Path("data/raw/images")
RAW_MASK_DIR = Path("data/raw/masks")
NNUNET_DATASET_DIR = Path("data/nnunet/nnUNet_raw/Dataset201_MyTask")
IMAGES_TR = NNUNET_DATASET_DIR / "imagesTr"
LABELS_TR = NNUNET_DATASET_DIR / "labelsTr"
IMAGES_TS = NNUNET_DATASET_DIR / "imagesTs"

def ensure_dirs():
    for d in [IMAGES_TR, LABELS_TR, IMAGES_TS]:
        d.mkdir(parents=True, exist_ok=True)

def copy_and_rename(src, dst):
    """Copy file from src to dst (overwrite if exists)."""
    shutil.copy(str(src), str(dst))

def binarize_mask(mask_path):
    """Load mask, binarize to 0/1, and overwrite."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    unique_vals_before = np.unique(data)
    print(f"[DEBUG] Before binarization, mask {mask_path.name} unique values: {unique_vals_before}")
    data_bin = (data > 0).astype(np.uint8)  # 所有非0变成1
    bin_img = nib.Nifti1Image(data_bin, img.affine, img.header)
    nib.save(bin_img, str(mask_path))

def check_mask_binary(mask_path):
    """Check that mask contains only 0 and 1."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    unique_vals = np.unique(data)
    valid = np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [0, 1])
    if not valid:
        print(f"[ERROR] Mask {mask_path.name} contains values other than 0 and 1: {unique_vals}")
    return valid

def main():
    ensure_dirs()
    cases = sorted([f.name.replace('.nii.gz', '') for f in RAW_IMAGE_DIR.glob("*.nii.gz")])
    print(f"Found {len(cases)} cases in {RAW_IMAGE_DIR}")
    kept, skipped = 0, 0

    for case in cases:
        print(f"Processing case: {case}")
        img_path = RAW_IMAGE_DIR / f"{case}.nii.gz"
        mask_path = RAW_MASK_DIR / f"{case}.nii.gz"
        if not img_path.exists() or not mask_path.exists():
            print(f"[WARNING] Missing image or mask for {case}, skipping.")
            skipped += 1
            continue

        # shape check
        img_shape = nib.load(str(img_path)).shape
        mask_shape = nib.load(str(mask_path)).shape
        if img_shape != mask_shape:
            print(f"[WARNING] Shape mismatch for {case}: image {img_shape}, mask {mask_shape}, skipping.")
            skipped += 1
            continue

        # 二值化mask，覆盖原文件
        binarize_mask(mask_path)

        # 检查二值化结果是否合规
        if not check_mask_binary(mask_path):
            print(f"[WARNING] Mask check failed for {case}, skipping.")
            skipped += 1
            continue

        # nnUNet expects images in the format case_0000.nii.gz
        img_dst = IMAGES_TR / f"{case}_0000.nii.gz"
        mask_dst = LABELS_TR / f"{case}.nii.gz"
        copy_and_rename(img_path, img_dst)
        copy_and_rename(mask_path, mask_dst)
        kept += 1
        print(f"[INFO] Copied {img_path.name} and {mask_path.name} to nnUNet imagesTr/labelsTr")

    print(f"All images and masks copied to {NNUNET_DATASET_DIR}")
    print(f"Kept: {kept}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
