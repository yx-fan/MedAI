import os, glob
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
    sizes = np.array(sizes)
    print("\n=== ROI size stats ===")
    print(f"Min: {sizes.min(axis=0)}")
    print(f"Median: {np.median(sizes, axis=0)}")
    print(f"Max: {sizes.max(axis=0)}")
    return sizes

if __name__ == "__main__":
    analyze_roi_size("./data/raw/masks")
