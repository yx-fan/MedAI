import os, glob
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, EnsureTyped, RandSpatialCropd, Lambdad,
    CenterSpatialCropd
)
from monai.data import Dataset


def get_dataloaders(data_dir="./data/raw", batch_size=4, debug=False):
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))

    if debug:
        # 调试时多取一些样本，避免 BN 报错
        images, labels = images[:8], labels[:8]
        batch_size = max(2, batch_size)

    n_train = int(0.8 * len(images))
    train_files = [{"image": i, "label": l} for i, l in zip(images[:n_train], labels[:n_train])]
    val_files   = [{"image": i, "label": l} for i, l in zip(images[n_train:], labels[n_train:])]

    # 论文输入大小 224×224（2D）
    roi_size = (224, 224, 1)

    # ---- Train transforms ----
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),   # image: [1,H,W,D], label: [1,H,W,D]
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
        # ⭐ 关键修复：把 D=1 的维度去掉
        Lambdad(keys="image", func=lambda x: x.squeeze(-1)),   # [1,H,W]
        Lambdad(keys="label", func=lambda x: x.squeeze(0).squeeze(-1).long()),  # [H,W]
    ])

    # ---- Val transforms ----
    # ✅ 验证集用中心裁剪，避免随机性导致指标不稳定
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0),
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        EnsureTyped(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: x.squeeze(-1)),   # [1,H,W]
        Lambdad(keys="label", func=lambda x: x.squeeze(0).squeeze(-1).long()),  # [H,W]
    ])

    # ---- Datasets / Loaders ----
    train_ds, val_ds = Dataset(train_files, train_transforms), Dataset(val_files, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader
