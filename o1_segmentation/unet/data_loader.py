import os, glob, random
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped, SpatialPadd
)
from monai.data import CacheDataset, Dataset, list_data_collate

def _pair_files(images: List[str], labels: List[str]) -> List[Dict[str, str]]:
    """pair image & label by去掉扩展名后的前缀（要求同名）"""
    img_map = {os.path.splitext(os.path.basename(p))[0].replace(".nii",""): p for p in images}
    lbl_map = {os.path.splitext(os.path.basename(p))[0].replace(".nii",""): p for p in labels}
    keys = sorted(list(set(img_map.keys()) & set(lbl_map.keys())))
    return [{"image": img_map[k], "label": lbl_map[k]} for k in keys]

def _split_cases(pairs: List[Dict[str,str]], train_ratio=0.8, seed=2024):
    random.Random(seed).shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    return pairs[:n_train], pairs[n_train:]

def get_dataloaders(
    data_dir: str = "./data/raw",
    batch_size: int = 4,
    debug: bool = False,
):
    """
    为直肠癌小体素分割准备的 DataLoader（与训练脚本严格对齐）
    - 训练：强正样本采样 + 随机增强
    - 验证：全图（仅重采样+归一化），与滑窗 roi_size 对齐
    """
    # -------------------------
    # 文件收集
    # -------------------------
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii*")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii*")))
    if debug:
        images, labels = images[:10], labels[:10]  # 小集快速跑通
    assert len(images) and len(labels), f"No files found in {data_dir}/images or {data_dir}/masks"
    pairs = _pair_files(images, labels)
    print(f"[DATA] paired cases: {len(pairs)}")

    # -------------------------
    # 病例级固定随机划分（8:2）
    # -------------------------
    train_files, val_files = _split_cases(pairs, train_ratio=0.8, seed=42)
    print(f"[SPLIT] train: {len(train_files)} | val: {len(val_files)}")

    # -------------------------
    # 形状/采样参数（与模型验证滑窗对齐）
    # -------------------------
    patch_train = (48, 48, 24) if debug else (160, 160, 96)
    pixdim = (1.0, 1.0, 3.0)  # 你之前用的 spacing
    # 直肠癌极少前景：提高正样本比例与采样次数
    pos_samples, neg_samples, num_samples = 4, 1, 4

    # -------------------------
    # 变换（Train）
    # -------------------------
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=5),
        # 关键：强正样本裁剪（前景极少）
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_train,
            pos=pos_samples,
            neg=neg_samples,
            num_samples=num_samples,
            image_key="image",
            allow_smaller=True,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_train),
        # 轻量增强（不破坏解剖一致性）
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])

    # -------------------------
    # 变换（Val，全图）
    # -------------------------
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    # -------------------------
    # 数据集（CacheDataset 加速 I/O 与重采样）
    # -------------------------
    # cache_rate 取个中庸值；debug 下加到 1.0 更快
    train_cache = 1.0 if debug else 0.2
    val_cache = 1.0 if debug else 0.5

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=train_cache, num_workers=0)
    val_ds   = CacheDataset(data=val_files,   transform=val_transforms,   cache_rate=val_cache,   num_workers=0)

    # -------------------------
    # DataLoader
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if debug else 4,
        pin_memory=True,
        persistent_workers=not debug,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,                 # 验证全图 → 逐例推理
        shuffle=False,
        num_workers=2 if debug else 4,
        pin_memory=True,
        persistent_workers=not debug,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader
