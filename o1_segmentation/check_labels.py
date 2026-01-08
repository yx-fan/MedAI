import os
import glob
import nibabel as nib
import numpy as np
from collections import defaultdict


def check_masks(data_dir="./data/raw/masks", max_files=None):
    mask_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    print(f"Found {len(mask_files)} mask files")
    
    if max_files:
        mask_files = mask_files[:max_files]
    
    ratios = []
    label_stats = defaultdict(lambda: {'count': 0, 'cases': 0})
    
    for i, f in enumerate(mask_files):
        mask = nib.load(f).get_fdata()
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        ratio = dict(zip(unique.astype(int), counts / total))
        
        foreground_ratio = ratio.get(1, 0.0)
        ratios.append(foreground_ratio)
        
        for label, count in zip(unique.astype(int), counts):
            label_stats[label]['count'] += count
            label_stats[label]['cases'] += 1
        
        if max_files and max_files <= 20:
            print(f"[{i}] {os.path.basename(f)}")
            print(f"    unique values: {dict(zip(unique.astype(int), counts))}")
            print(f"    foreground ratio: {foreground_ratio:.6f}")
    
    if ratios:
        print("\n=== Summary ===")
        print(f"Avg foreground ratio: {np.mean(ratios):.6f}")
        print(f"Min: {np.min(ratios):.6f}, Max: {np.max(ratios):.6f}")
        print(f"Median: {np.median(ratios):.6f}")
        
        print("\n=== Label Distribution ===")
        total_voxels = sum(s['count'] for s in label_stats.values())
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            ratio = stats['count'] / total_voxels if total_voxels > 0 else 0
            print(f"Label {label}: {stats['count']:,} voxels ({ratio*100:.2f}%), "
                  f"in {stats['cases']} cases")


if __name__ == "__main__":
    check_masks()
