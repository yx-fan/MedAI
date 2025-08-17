import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Pick a sample patient ID
pid = "160980"
ct_path = Path("data/raw/images") / f"{pid}.nii.gz"
mask_path = Path("data/raw/masks") / f"{pid}.nii.gz"

# Load CT and mask
ct_img = nib.load(str(ct_path)).get_fdata()
mask_img = nib.load(str(mask_path)).get_fdata()

# Select a few slices with tumors
tumor_slices = np.where(np.any(mask_img > 0, axis=(0, 1)))[0]
sample_slices = tumor_slices[len(tumor_slices)//2 - 1 : len(tumor_slices)//2 + 2]  # Three slices

# Save images
for i, idx in enumerate(sample_slices):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(ct_img[:, :, idx], cmap="gray")
    plt.title(f"CT Slice {idx}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(ct_img[:, :, idx], cmap="gray")
    plt.imshow(mask_img[:, :, idx], alpha=0.3, cmap="Reds")
    plt.title(f"CT + Mask Slice {idx}")
    plt.axis("off")

    out_path = f"sample_{pid}_{idx}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {out_path}")
