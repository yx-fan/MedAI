import os
import glob
import nibabel as nib
import numpy as np


def analyze_intensity_range(data_dir="./data/raw/images"):
    nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not nii_files:
        print(f"No .nii.gz files found in {data_dir}")
        return
    
    all_mins, all_maxs, all_means = [], [], []
    
    for f in nii_files:
        try:
            img = nib.load(f)
            data = img.get_fdata().astype(np.float32).flatten()
            
            case_min, case_max = data.min(), data.max()
            case_mean = data.mean()
            all_mins.append(case_min)
            all_maxs.append(case_max)
            all_means.append(case_mean)
            
            print(f"{os.path.basename(f)} -> min={case_min:.1f}, max={case_max:.1f}, mean={case_mean:.1f}")
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    
    if not all_mins:
        return
    
    all_mins = np.array(all_mins)
    all_maxs = np.array(all_maxs)
    all_means = np.array(all_means)
    
    print("\n=== Intensity Range Statistics ===")
    print(f"Number of cases: {len(nii_files)}")
    print(f"Global min: {all_mins.min():.1f}, Global max: {all_maxs.max():.1f}")
    print(f"Median min: {np.median(all_mins):.1f}, Median max: {np.median(all_maxs):.1f}")
    print(f"Mean intensity: {all_means.mean():.1f} ± {all_means.std():.1f}")
    print(f"\nRecommended normalization range:")
    print(f"  a_min={int(np.percentile(all_mins, 5)):.0f} (5th percentile)")
    print(f"  a_max={int(np.percentile(all_maxs, 95)):.0f} (95th percentile)")
    
    return all_mins, all_maxs


if __name__ == "__main__":
    analyze_intensity_range("./data/raw/images")
