#!/usr/bin/env python3
"""
检查验证数据的尺寸，确保它们能被UNet处理
UNet需要输入尺寸能被16整除（4个下采样层，每个stride=2）
"""
import os
import glob
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    EnsureTyped,
    DivisiblePadd,
)

def check_val_data_shapes(data_dir="./data/raw"):
    """检查验证数据的尺寸分布"""
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    
    n_train = int(0.8 * len(images))
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(images[n_train:], labels[n_train:])]
    
    # 使用与验证时相同的transforms（包括DivisiblePadd）
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=16),  # UNet requires dimensions divisible by 16
    ])
    
    shapes = []
    problematic = []
    
    print(f"检查 {len(val_files)} 个验证样本...")
    print("-" * 80)
    
    for i, file_dict in enumerate(val_transforms(val_files)):
        img_shape = file_dict["image"].shape[1:]  # 去掉channel维度
        label_shape = file_dict["label"].shape[1:]
        
        # 检查是否能被16整除
        divisible_by_16 = all(s % 16 == 0 for s in img_shape)
        
        shapes.append(img_shape)
        
        if not divisible_by_16 or img_shape != label_shape:
            problematic.append({
                "idx": i,
                "image": val_files[i]["image"],
                "shape": img_shape,
                "label_shape": label_shape,
                "divisible_by_16": divisible_by_16,
                "padding_needed": tuple((16 - (s % 16)) % 16 for s in img_shape)
            })
    
    print(f"\n总样本数: {len(shapes)}")
    print(f"问题样本数: {len(problematic)}")
    
    if shapes:
        shapes_arr = np.array(shapes)
        print(f"\n尺寸统计:")
        print(f"  X: min={shapes_arr[:, 0].min()}, max={shapes_arr[:, 0].max()}, mean={shapes_arr[:, 0].mean():.1f}")
        print(f"  Y: min={shapes_arr[:, 1].min()}, max={shapes_arr[:, 1].max()}, mean={shapes_arr[:, 1].mean():.1f}")
        print(f"  Z: min={shapes_arr[:, 2].min()}, max={shapes_arr[:, 2].max()}, mean={shapes_arr[:, 2].mean():.1f}")
    
    if problematic:
        print(f"\n⚠️  发现 {len(problematic)} 个问题样本:")
        for p in problematic[:10]:  # 只显示前10个
            print(f"  [{p['idx']}] {os.path.basename(p['image'])}")
            print(f"      形状: {p['shape']}")
            print(f"      能被16整除: {p['divisible_by_16']}")
            print(f"      需要padding: {p['padding_needed']}")
        if len(problematic) > 10:
            print(f"  ... 还有 {len(problematic) - 10} 个问题样本")
    else:
        print("\n✅ 所有样本尺寸都能被16整除")
    
    return problematic

if __name__ == "__main__":
    problematic = check_val_data_shapes()
    
    if problematic:
        print("\n⚠️  仍有问题样本，请检查数据")
    else:
        print("\n✅ 所有验证数据已正确padding，可以被UNet处理")
