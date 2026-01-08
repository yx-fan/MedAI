import os
import glob
import nibabel as nib
import numpy as np


def analyze_roi_size(label_dir="./data/raw/masks"):
    sizes = []
    for f in sorted(glob.glob(os.path.join(label_dir, "*.nii.gz"))):
        mask = nib.load(f).get_fdata().astype(np.uint8)
        if mask.sum() == 0:
            continue
        coords = np.argwhere(mask > 0)
        minz, miny, minx = coords.min(axis=0)
        maxz, maxy, maxx = coords.max(axis=0)
        dz, dy, dx = (maxz - minz + 1, maxy - miny + 1, maxx - minx + 1)
        sizes.append((dx, dy, dz))
        print(f"{os.path.basename(f)} -> ROI size: {dx}×{dy}×{dz}")
    
    if not sizes:
        print("No valid ROIs found")
        return
    
    sizes = np.array(sizes)
    print("\n=== ROI Size Statistics ===")
    print(f"Total cases: {len(sizes)}")
    print(f"X: min={sizes[:, 0].min()}, median={np.median(sizes[:, 0]):.0f}, max={sizes[:, 0].max()}")
    print(f"Y: min={sizes[:, 1].min()}, median={np.median(sizes[:, 1]):.0f}, max={sizes[:, 1].max()}")
    print(f"Z: min={sizes[:, 2].min()}, median={np.median(sizes[:, 2]):.0f}, max={sizes[:, 2].max()}")
    
    return sizes


if __name__ == "__main__":
    analyze_roi_size("./data/raw/masks")
