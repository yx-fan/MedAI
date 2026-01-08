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
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
    DivisiblePadd,
)
from monai.data import Dataset, list_data_collate


def get_dataloaders(data_dir="./data/raw", batch_size=2, patch_size=(160, 160, 80), debug=False):
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    print("Found images:", len(images))
    print("Found labels:", len(labels))

    if debug:
        images = images[:8]
        labels = labels[:8]
        print(f"[DEBUG] Using subset: {len(images)} images")

    n_train = int(0.8 * len(images))
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(images[:n_train], labels[:n_train])]
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images[n_train:], labels[n_train:])]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=6, neg=1,  # Increased pos from 4 to 6 for better foreground sampling
            num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=(-0.1, 0.1)),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=32),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2 if debug else 2,
        pin_memory=False,
        persistent_workers=not debug,
        prefetch_factor=1 if not debug else None,
        collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2 if debug else 2,
        pin_memory=False,
        persistent_workers=not debug,
        prefetch_factor=1 if not debug else None,
        collate_fn=list_data_collate
    )

    return train_loader, val_loader
