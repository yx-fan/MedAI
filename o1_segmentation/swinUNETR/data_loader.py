import os
import glob
import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandSpatialCropd,
    RandFlipd,
    EnsureTyped,
    Resized,
)
from monai.data import Dataset


# ✅ 定义一个函数：返回 train_loader 和 val_loader
def get_dataloaders(data_dir="../../data/raw", batch_size=2, patch_size=(96, 96, 96)):
    # 获取所有 NIfTI 文件
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    print("Found images:", len(images))
    print("Found labels:", len(labels))

    # 划分 train / val（简单 80/20）
    n_train = int(0.8 * len(images))
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(images[:n_train], labels[:n_train])]
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images[n_train:], labels[n_train:])]

    # MONAI transform
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        Resized(keys=["image", "label"], spatial_size=patch_size, mode=("trilinear", "nearest")),  # ✅ 保证对齐
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=patch_size, mode=("trilinear", "nearest")),  # ✅ 保证对齐
        EnsureTyped(keys=["image", "label"]),
    ])

    # Dataset
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ⚡ 修改：返回 Tensor 而不是 dict
    def to_tensor_loader(loader):
        for batch in loader:
            yield batch["image"], batch["label"]

    return to_tensor_loader(train_loader), to_tensor_loader(val_loader)


# ✅ 测试一下 dataloader
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    images, masks = next(iter(train_loader))
    print(type(images), images.shape)
    print(type(masks), masks.shape)
