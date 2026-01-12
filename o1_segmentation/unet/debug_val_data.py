#!/usr/bin/env python3
"""
调试脚本：模拟训练时的验证数据流，检查是否有问题
"""
import os
import glob
import random
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    EnsureTyped,
    DivisiblePadd,
)
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.nets import UNet

def debug_val_data(data_dir="./data/raw", random_seed=42):
    """模拟训练时的验证数据流"""
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    
    # 使用与data_loader.py相同的分割逻辑
    pairs = list(zip(images, labels))
    random.seed(random_seed)
    random.shuffle(pairs)
    
    n_train = int(0.8 * len(pairs))
    val_pairs = pairs[n_train:]
    val_files = [{"image": img, "label": lbl} for img, lbl in val_pairs]
    
    print(f"验证集样本数: {len(val_files)}")
    print(f"使用与训练时相同的transforms...\n")
    
    # 使用与data_loader.py完全相同的val_transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=16),
    ])
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0,  # 使用0避免多进程问题
        collate_fn=list_data_collate
    )
    
    # 创建一个简单的UNet来测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.eval()
    
    print("开始测试验证数据流...")
    print("-" * 80)
    
    problematic = []
    
    for step, batch in enumerate(val_loader):
        try:
            images = batch["image"].to(device, non_blocking=False)
            masks = batch["label"].to(device, non_blocking=False).long()
            
            # 检查尺寸（只检查空间维度，不包括batch和channel）
            img_shape = images.shape[2:]  # 去掉batch和channel维度，只保留空间维度
            print(f"Step {step}: image shape = {images.shape}, spatial = {img_shape}")
            
            # 检查是否能被16整除（只检查空间维度）
            divisible_by_16 = all(s % 16 == 0 for s in img_shape)
            if divisible_by_16:
                print(f"  ✅ 尺寸能被16整除")
            else:
                print(f"  ⚠️  尺寸不能被16整除! (需要padding到: {tuple((s + 15) // 16 * 16 for s in img_shape)})")
            if not divisible_by_16:
                problematic.append({
                    "step": step,
                    "file": val_files[step]["image"],
                    "shape": img_shape,
                    "divisible": divisible_by_16
                })
            
            # 尝试forward pass
            with torch.inference_mode():
                outputs = model(images)
                print(f"  ✅ Forward pass成功, output shape = {outputs.shape}")
                
        except Exception as e:
            problematic.append({
                "step": step,
                "file": val_files[step]["image"],
                "error": str(e)
            })
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "-" * 80)
    if problematic:
        print(f"\n⚠️  发现 {len(problematic)} 个问题样本:")
        for p in problematic:
            print(f"  Step {p['step']}: {os.path.basename(p['file'])}")
            if "error" in p:
                print(f"    错误: {p['error']}")
            else:
                print(f"    形状: {p['shape']}, 能被16整除: {p['divisible']}")
    else:
        print("\n✅ 所有验证样本都能正常处理")
    
    return problematic

if __name__ == "__main__":
    problematic = debug_val_data()
