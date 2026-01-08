import os
import glob
import nibabel as nib
import numpy as np


def analyze_spacing(data_dir="./data/raw/images"):
    nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not nii_files:
        print(f"No .nii.gz files found in {data_dir}")
        return
    
    spacings = []
    for f in nii_files:
        try:
            img = nib.load(f)
            zooms = img.header.get_zooms()[:3]
            spacings.append(zooms)
            print(f"{os.path.basename(f)} spacing: {zooms}")
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    
    if not spacings:
        return
    
    spacings = np.array(spacings)
    print("\n=== Spacing Statistics ===")
    print(f"Number of cases: {len(spacings)}")
    print(f"X spacing: min={spacings[:,0].min():.3f}, median={np.median(spacings[:,0]):.3f}, max={spacings[:,0].max():.3f}")
    print(f"Y spacing: min={spacings[:,1].min():.3f}, median={np.median(spacings[:,1]):.3f}, max={spacings[:,1].max():.3f}")
    print(f"Z spacing: min={spacings[:,2].min():.3f}, median={np.median(spacings[:,2]):.3f}, max={spacings[:,2].max():.3f}")
    
    unique_spacings = {tuple(s) for s in spacings}
    if len(unique_spacings) > 1:
        print(f"\n⚠️  Multiple spacings detected: {len(unique_spacings)} unique spacings")
        from collections import defaultdict
        spacing_counts = defaultdict(int)
        for s in spacings:
            spacing_counts[tuple(s)] += 1
        print("Most common spacings:")
        for spacing, count in sorted(spacing_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {spacing}: {count} cases")
    
    return spacings


if __name__ == "__main__":
    analyze_spacing("./data/raw/images")
