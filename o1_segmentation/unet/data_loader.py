import os
import glob
import random
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


def get_dataloaders(data_dir="./data/raw", batch_size=2, patch_size=(160, 160, 96), debug=False, random_seed=42):
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    print("Found images:", len(images))
    print("Found labels:", len(labels))

    if debug:
        images = images[:8]
        labels = labels[:8]
        print(f"[DEBUG] Using subset: {len(images)} images")

    # 随机分割训练/验证集（使用固定seed保证可复现）
    pairs = list(zip(images, labels))
    random.seed(random_seed)
    random.shuffle(pairs)
    
    n_train = int(0.8 * len(pairs))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    
    train_files = [{"image": img, "label": lbl} for img, lbl in train_pairs]
    val_files = [{"image": img, "label": lbl} for img, lbl in val_pairs]
    
    print(f"Train/Val split: {len(train_files)}/{len(val_files)} (random seed={random_seed})")

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=6, neg=1,
            num_samples=3,  # Reduced from 4 to 3 for faster training
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(0.9, 1.1)),
        RandShiftIntensityd(keys=["image"], prob=0.25, offsets=(-0.05, 0.05)),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=16),  # UNet downsampling factor is 16 (2^4)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=16),  # UNet requires dimensions divisible by 16
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2 if debug else 4,  # Reduced to avoid memory issues
        pin_memory=False,
        persistent_workers=not debug,
        prefetch_factor=2 if not debug else None,
        collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2 if debug else 4,  # Reduced to avoid memory issues
        pin_memory=False,
        persistent_workers=not debug,
        prefetch_factor=2 if not debug else None,
        collate_fn=list_data_collate
    )

    return train_loader, val_loader
