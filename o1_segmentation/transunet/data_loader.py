import os
import glob
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    DivisiblePadd,
)
from monai.data import Dataset, list_data_collate


def get_transunet_dataloaders(data_dir="./data/raw", batch_size=2, patch_size=(160, 160, 128), debug=False):
    """
    Create train and validation dataloaders for TransUNet 3D segmentation.
    Uses the same preprocessing and augmentation pipeline as the UNet version.
    Args:
        data_dir: path to dataset, should contain 'images' and 'masks' subdirs
        batch_size: batch size for training
        patch_size: crop size for training patches
        debug: if True, only load a small subset of data for quick testing
    """
    # Collect files
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    print(f"[TransUNet] Found {len(images)} images and {len(labels)} labels")

    if debug:
        images = images[:8]
        labels = labels[:8]
        print(f"[DEBUG] Using subset of {len(images)} samples for TransUNet")

    # Train/validation split (80/20)
    n_train = int(0.8 * len(images))
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(images[:n_train], labels[:n_train])]
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images[n_train:], labels[n_train:])]

    # Training transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=4, neg=1,
            num_samples=8,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=32),
    ])

    # Validation transforms (no augmentation)
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if debug else 4,
        pin_memory=False,
        persistent_workers=not debug,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2 if debug else 4,
        pin_memory=False,
        persistent_workers=not debug,
        collate_fn=list_data_collate,
    )

    print(f"[TransUNet] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader
