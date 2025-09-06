import os, glob
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, EnsureTyped, RandSpatialCropd, CenterSpatialCropd, Lambdad
)
from monai.data import Dataset
import numpy as np
import torch
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloaders(data_dir="./data/raw", batch_size=4, debug=False):
    set_seed(42)

    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks",  "*.nii.gz")))

    if debug:
        images, labels = images[:8], labels[:8]
        batch_size = max(2, batch_size)

    n_train = int(0.8 * len(images))
    train_files = [{"image": i, "label": l} for i, l in zip(images[:n_train], labels[:n_train])]
    val_files   = [{"image": i, "label": l} for i, l in zip(images[n_train:], labels[n_train:])]

    roi_size = (224, 224, 1)

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: x),           # [1,H,W]
        Lambdad(keys="label", func=lambda x: x[0].long()), # [H,W] int
    ])

    # ⭐ 验证用“中心裁剪”，去掉随机性
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0),
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        EnsureTyped(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: x),           # [1,H,W]
        Lambdad(keys="label", func=lambda x: x[0].long()), # [H,W] int
    ])

    train_ds, val_ds = Dataset(train_files, train_transforms), Dataset(val_files, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1,        shuffle=False, num_workers=4)

    return train_loader, val_loader
