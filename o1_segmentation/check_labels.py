import os
import glob
import nibabel as nib
import numpy as np

def check_masks(data_dir="./data/raw/masks", max_files=10):
    mask_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    print(f"Found {len(mask_files)} mask files")

    ratios = []
    for i, f in enumerate(mask_files[:max_files]):  # 只检查前 max_files 个
        mask = nib.load(f).get_fdata()
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        ratio = dict(zip(unique.astype(int), counts / total))

        ratios.append(ratio.get(1, 0.0))  # foreground 占比
        print(f"[{i}] {os.path.basename(f)}")
        print(f"    unique values: {dict(zip(unique.astype(int), counts))}")
        print(f"    foreground ratio: {ratio.get(1, 0.0):.6f}")

    if ratios:
        print("\n=== Summary ===")
        print(f"Avg foreground ratio (前景平均占比): {np.mean(ratios):.6f}")
        print(f"Min foreground ratio: {np.min(ratios):.6f}")
        print(f"Max foreground ratio: {np.max(ratios):.6f}")

if __name__ == "__main__":
    check_masks()
