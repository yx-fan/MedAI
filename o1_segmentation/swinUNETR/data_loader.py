import os
import glob
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


def get_dataloaders(data_dir="./data/raw", batch_size=2, patch_size=(96, 96, 96)):
    """
    Create train and validation dataloaders for medical image segmentation.
    
    Args:
        data_dir (str): path to dataset folder. Expecting `images/` and `masks/`.
        batch_size (int): batch size.
        patch_size (tuple): size of cropped/resized patches.

    Returns:
        train_loader, val_loader (torch.utils.data.DataLoader)
    """
    # Collect all image/mask files
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    print("Found images:", len(images))
    print("Found labels:", len(labels))

    # Train/validation split (80/20)
    n_train = int(0.8 * len(images))
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(images[:n_train], labels[:n_train])]
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images[n_train:], labels[n_train:])]

    # Training transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        Resized(keys=["image", "label"], spatial_size=patch_size, mode=("trilinear", "nearest")),  # keep aligned
        EnsureTyped(keys=["image", "label"]),
    ])

    # Validation transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=patch_size, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


# ✅ Quick test
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    batch = next(iter(train_loader))
    images, masks = batch["image"], batch["label"]
    print("Image batch:", images.shape)
    print("Mask batch:", masks.shape)
