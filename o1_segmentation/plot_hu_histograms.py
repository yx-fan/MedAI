import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def plot_hu_histograms(data_dir="./data/raw/images", num_cases=3, hu_range=(-1000, 1000)):
    """
    Plot HU histograms of randomly selected NIfTI cases and print stats.
    
    Args:
        data_dir (str): Directory containing CT NIfTI files.
        num_cases (int): Number of cases to plot.
        hu_range (tuple): Range of HU values to display in histogram.
    """
    nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not nii_files:
        print(f"❌ No .nii.gz files found in {data_dir}")
        return

    # Randomly sample cases
    sampled_files = np.random.choice(nii_files, size=min(num_cases, len(nii_files)), replace=False)

    plt.figure(figsize=(15, 4))
    for i, f in enumerate(sampled_files, 1):
        img = nib.load(f)
        data = img.get_fdata().astype(np.float32).flatten()

        # Print basic statistics
        print(f"\n📊 {os.path.basename(f)}")
        print(f"  Min: {data.min():.1f}, Max: {data.max():.1f}")
        print(f"  Mean: {data.mean():.1f}, Median: {np.median(data):.1f}")
        print(f"  1st percentile: {np.percentile(data, 1):.1f}")
        print(f"  99th percentile: {np.percentile(data, 99):.1f}")
        print(f"  Unique values (sample): {np.unique(data)[:10]} ... total={len(np.unique(data))}")

        # Plot histogram
        plt.subplot(1, num_cases, i)
        plt.hist(data, bins=200, range=hu_range, color="steelblue", alpha=0.7)
        plt.title(os.path.basename(f), fontsize=10)
        plt.xlabel("HU")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_hu_histograms("./data/raw/images", num_cases=3, hu_range=(-500, 500))
