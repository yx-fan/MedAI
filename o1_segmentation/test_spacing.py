import nibabel as nib
import matplotlib.pyplot as plt
from monai.transforms import Spacing, EnsureChannelFirst, ScaleIntensityRange

# Load case
path = "./data/raw/images/160980.nii.gz"
img = nib.load(path)
data = img.get_fdata()

print("Original shape:", data.shape, "spacing:", img.header.get_zooms())

# MONAI transform
resample_to_111 = Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear")
resample_to_113 = Spacing(pixdim=(1.0, 1.0, 3.0), mode="bilinear")

# Add channel for MONAI
x = data[None]  # shape: (1, H, W, D)

# Resample
x_111 = resample_to_111(x)
x_113 = resample_to_113(x)

print("Resampled (1,1,1):", x_111.shape)
print("Resampled (1,1,3):", x_113.shape)

# Pick the same slice (e.g., middle slice)
mid_slice = data.shape[2] // 2

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(x_111[0,:,:,mid_slice], cmap="gray")
plt.title("Spacing (1,1,1)")

plt.subplot(1,2,2)
plt.imshow(x_113[0,:,:,mid_slice//3], cmap="gray")  # adjust index since z is downsampled
plt.title("Spacing (1,1,3)")

plt.show()
