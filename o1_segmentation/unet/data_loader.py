import os
import glob
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
    DivisiblePadd,
)
from monai.data import Dataset, list_data_collate

def get_dataloaders(data_dir="./data/raw", batch_size=2, patch_size=(192, 192, 96), debug=False):
    """
    Create train and validation dataloaders for rectal cancer CT segmentation.
    Args:
        data_dir: path to dataset, should contain 'images' and 'masks' subdirs
        batch_size: batch size for training
        patch_size: crop size for training patches
        debug: if True, only load a small subset of data for quick testing
    """
    # Collect files
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    print("Found images:", len(images))
    print("Found labels:", len(labels))

    if debug:  # Use a small subset for debugging
        images = images[:8]
        labels = labels[:8]
        print(f"[DEBUG] Using subset: {len(images)} images")

    # Train/validation split (80/20)
    n_train = int(0.8 * len(images))
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(images[:n_train], labels[:n_train])]
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images[n_train:], labels[n_train:])]
    # Example train_files: [{"image": ".../images/case_0001.nii.gz", "label": ".../masks/case_0001.nii.gz"}, ...]
    # Example val_files: [{"image": ".../images/case_0005.nii.gz", "label": ".../masks/case_0005.nii.gz"}, ...]

    # Training transforms (enhanced with more augmentations)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=4, neg=1,
            num_samples=4,  # Reduced from 8 to 4 for faster training (still provides good diversity)
        ),
        # Geometric augmentations (optimized for speed)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),  # Flip along all axes
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),          # Rotate 90, 180, or 270 degrees
        # Removed RandRotated and RandZoomd - these 3D operations are very slow
        # Can re-enable if needed, but will significantly slow down training
        # Intensity augmentations (kept - these are fast)
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        # Removed RandGaussianSmoothd - slow 3D operation
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=(-0.1, 0.1)),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=32),
    ])

    # Validation transforms (No cropping or augmentation, just normalization)
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2 if debug else 6,   # Reduced to 6 to avoid pin_memory issues
        pin_memory=False,  # Disabled due to multiprocessing compatibility issues
        persistent_workers=not debug,
        prefetch_factor=2 if not debug else None,  # Prefetch batches for faster loading
        collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,  # Increased to 4 for faster validation (was 2)
        num_workers=2 if debug else 6,  # Reduced to 6 to avoid pin_memory issues
        pin_memory=False,  # Disabled due to multiprocessing compatibility issues
        persistent_workers=not debug,
        prefetch_factor=2 if not debug else None,
        collate_fn=list_data_collate
    )

    return train_loader, val_loader
