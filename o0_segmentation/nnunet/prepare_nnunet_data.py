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
LABELS_TS = NNUNET_DATASET_DIR / "labelsTs"

TEST_FRACTION = 0.10
RANDOM_SEED = 42

def ensure_dirs():
    for d in [IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS]:
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
    data_bin = (data > 0).astype(np.uint8)  # Convert to binary (0 or 1)
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

def collect_valid_cases():
    """Collect valid cases for training."""
    cases = sorted([f.name.replace('.nii.gz', '') for f in RAW_IMAGE_DIR.glob("*.nii.gz")])
    valid = []
    missing = 0
    mismatched = 0
    for case in cases:
        img_path = RAW_IMAGE_DIR / f"{case}.nii.gz"
        mask_path = RAW_MASK_DIR / f"{case}.nii.gz"
        if not img_path.exists() or not mask_path.exists():
            missing += 1
            continue
        try:
            img_shape = nib.load(str(img_path)).shape
            mask_shape = nib.load(str(mask_path)).shape
        except Exception as e:
            print(f"[WARNING] Failed to load {case}: {e}")
            continue
        if img_shape != mask_shape:
            mismatched += 1
            continue
        valid.append(case)
    print(f"[INFO] Found {len(cases)} cases, valid={len(valid)}, missing={missing}, shape_mismatch={mismatched}")
    return valid

def process_case(case, img_dir, mask_dir):
    """Process a single case: binarize mask, check, and copy to target dirs."""
    img_path = RAW_IMAGE_DIR / f"{case}.nii.gz"
    mask_path = RAW_MASK_DIR / f"{case}.nii.gz"

    # binarize the mask
    binarize_mask(mask_path)
    if not check_mask_binary(mask_path):
        print(f"[WARNING] Mask check failed for {case}, skipping.")
        return False

    # Copy to target directories
    img_dst = img_dir / f"{case}_0000.nii.gz"
    mask_dst = mask_dir / f"{case}.nii.gz"

    copy_and_rename(img_path, img_dst)
    copy_and_rename(mask_path, mask_dst)
    print(f"[INFO] Copied {img_path.name} -> {img_dst}")
    print(f"[INFO] Copied {mask_path.name} -> {mask_dst}")
    return True

def main():
    ensure_dirs()

    valid_cases = collect_valid_cases()
    if len(valid_cases) == 0:
        print("[ERROR] No valid cases to process.")
        return

    np.random.seed(RANDOM_SEED)
    perm = np.random.permutation(len(valid_cases))
    test_count = max(1, int(round(len(valid_cases) * TEST_FRACTION)))
    test_idx = set(perm[:test_count])
    test_cases = [valid_cases[i] for i in test_idx]
    train_cases = [valid_cases[i] for i in perm if i not in test_idx]

    print(f"[SPLIT] Train: {len(train_cases)} cases, Test: {len(test_cases)} cases (test_fraction={TEST_FRACTION:.2f})")

    kept_tr = skipped_tr = 0
    for case in train_cases:
        ok = process_case(case, IMAGES_TR, LABELS_TR)
        if ok: kept_tr += 1
        else: skipped_tr += 1

    kept_ts = skipped_ts = 0
    for case in test_cases:
        ok = process_case(case, IMAGES_TS, LABELS_TS)
        if ok: kept_ts += 1
        else: skipped_ts += 1

    print(f"[SUMMARY] Copied to {NNUNET_DATASET_DIR}")
    print(f"[TRAIN] kept={kept_tr}, skipped={skipped_tr}")
    print(f"[TEST ] kept={kept_ts}, skipped={skipped_ts}")

if __name__ == "__main__":
    main()