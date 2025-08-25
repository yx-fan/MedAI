import os
import glob
import nibabel as nib
import numpy as np

def analyze_intensity_range(data_dir="./data/raw/images"):
    nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not nii_files:
        print(f"No .nii.gz files found in {data_dir}")
        return

    all_mins, all_maxs = [], []

    for f in nii_files:
        img = nib.load(f)
        data = img.get_fdata().astype(np.float32).flatten()

        case_min, case_max = data.min(), data.max()
        all_mins.append(case_min)
        all_maxs.append(case_max)

    all_mins = np.array(all_mins)
    all_maxs = np.array(all_maxs)

    print("\n=== Intensity Range Statistics ===")
    print(f"Number of cases: {len(nii_files)}")
    print(f"Global min: {all_mins.min():.1f}, Global max: {all_maxs.max():.1f}")
    print(f"Median min: {np.median(all_mins):.1f}, Median max: {np.median(all_maxs):.1f}")
    print(f"Unique min values: {np.unique(all_mins)}")
    print(f"Unique max values: {np.unique(all_maxs)}")

    return all_mins, all_maxs

if __name__ == "__main__":
    analyze_intensity_range("./data/raw/images")
