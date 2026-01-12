#!/usr/bin/env python3
"""
检查所有数据的尺寸，确保它们能被UNet处理
UNet需要输入尺寸能被16整除（4个下采样层，每个stride=2）

注意：虽然训练集使用RandCropByPosNegLabeld会裁剪成固定patch，但验证集使用完整图像，
所以需要确保所有完整图像在padding后都能被UNet处理。
"""
import os
import glob
import random
import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    EnsureTyped,
    DivisiblePadd,
)

def check_all_data_shapes(data_dir="./data/raw", check_all=True, random_seed=42):
    """
    检查数据的尺寸分布
    
    Args:
        check_all: 如果True，检查所有数据；如果False，只检查验证集
        random_seed: 用于数据分割的随机种子（与data_loader.py保持一致）
    """
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii.gz")))
    
    total = len(images)
    
    if check_all:
        # 检查所有数据
        files_to_check = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
        print(f"检查所有 {total} 个样本...")
    else:
        # 只检查验证集（使用与data_loader.py相同的分割逻辑）
        pairs = list(zip(images, labels))
        random.seed(random_seed)
        random.shuffle(pairs)
        
        n_train = int(0.8 * total)
        n_val = total - n_train
        val_pairs = pairs[n_train:]
        files_to_check = [{"image": img, "label": lbl} for img, lbl in val_pairs]
        
        print(f"数据分割信息 (seed={random_seed}):")
        print(f"  总样本数: {total}")
        print(f"  训练集: {n_train} ({n_train/total*100:.1f}%)")
        print(f"  验证集: {n_val} ({n_val/total*100:.1f}%)")
        print(f"\n检查 {len(files_to_check)} 个验证样本...")
    
    # 使用与验证时相同的transforms（包括DivisiblePadd）
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=94, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=16),  # UNet requires dimensions divisible by 16
    ])
    
    shapes = []
    problematic = []
    
    print("-" * 80)
    print("开始检查...\n")
    
    # 使用tqdm显示进度
    for i, file_pair in enumerate(tqdm(files_to_check, desc="检查数据", unit="样本")):
        try:
            # 对单个文件应用transforms
            file_dict = transforms(file_pair)
            img_shape = file_dict["image"].shape[1:]  # 去掉channel维度
            label_shape = file_dict["label"].shape[1:]
            
            # 检查是否能被16整除
            divisible_by_16 = all(s % 16 == 0 for s in img_shape)
            
            shapes.append(img_shape)
            
            if not divisible_by_16 or img_shape != label_shape:
                problematic.append({
                    "idx": i,
                    "image": file_pair["image"],
                    "shape": img_shape,
                    "label_shape": label_shape,
                    "divisible_by_16": divisible_by_16,
                    "padding_needed": tuple((16 - (s % 16)) % 16 for s in img_shape)
                })
        except Exception as e:
            # 如果某个文件处理失败，记录错误
            problematic.append({
                "idx": i,
                "image": file_pair["image"],
                "shape": None,
                "label_shape": None,
                "divisible_by_16": False,
                "padding_needed": None,
                "error": str(e)
            })
            print(f"\n⚠️  处理文件失败: {os.path.basename(file_pair['image'])} - {e}")
    
    print()  # 空行分隔
    
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
            if "error" in p:
                print(f"      错误: {p['error']}")
            else:
                print(f"      形状: {p['shape']}")
                print(f"      能被16整除: {p['divisible_by_16']}")
                if p['padding_needed']:
                    print(f"      需要padding: {p['padding_needed']}")
        if len(problematic) > 10:
            print(f"  ... 还有 {len(problematic) - 10} 个问题样本")
    else:
        print("\n✅ 所有样本尺寸都能被16整除")
    
    return problematic

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Check all data instead of just validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    
    problematic = check_all_data_shapes(check_all=args.all, random_seed=args.seed)
    
    if problematic:
        print("\n⚠️  仍有问题样本，请检查数据")
    else:
        if args.all:
            print("\n✅ 所有数据已正确padding，可以被UNet处理")
        else:
            print("\n✅ 所有验证数据已正确padding，可以被UNet处理")
