import os
import glob
import nibabel as nib
import numpy as np

def analyze_spacing(data_dir="./data/raw/images"):
    """
    Scan NIfTI files and compute the distribution of voxel spacing (mm).
    
    Args:
        data_dir (str): Directory containing CT NIfTI files.
    """
    # Collect all NIfTI files (*.nii.gz) from the directory
    nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not nii_files:
        print(f"❌ No .nii.gz files found in {data_dir}")
        return

    spacings = []
    for f in nii_files:
        try:
            # Load NIfTI image
            img = nib.load(f)
            # Extract voxel spacing (x, y, z)
            zooms = img.header.get_zooms()[:3]
            spacings.append(zooms)
            print(f"{os.path.basename(f)} spacing: {zooms}")
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}")

    # Convert to numpy array for easier statistics
    spacings = np.array(spacings)

    # Print summary statistics for each dimension
    print("\n=== Spacing Statistics ===")
    print(f"Number of cases: {len(spacings)}")
    print(f"X spacing: min={spacings[:,0].min():.2f}, median={np.median(spacings[:,0]):.2f}, max={spacings[:,0].max():.2f}")
    print(f"Y spacing: min={spacings[:,1].min():.2f}, median={np.median(spacings[:,1]):.2f}, max={spacings[:,1].max():.2f}")
    print(f"Z spacing: min={spacings[:,2].min():.2f}, median={np.median(spacings[:,2]):.2f}, max={spacings[:,2].max():.2f}")

    return spacings

if __name__ == "__main__":
    analyze_spacing("./data/raw/images")
