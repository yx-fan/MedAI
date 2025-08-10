import os
import nibabel as nib
import numpy as np
from pathlib import Path

# ===== Config =====
pred_dirs = {
    "2d": "data/nnunet/predictions/2d_pred",
    "3d_lowres": "data/nnunet/predictions/3d_lowres_pred",
    "3d_fullres": "data/nnunet/predictions/3d_fullres_pred"
}

# 权重（可以用测试集平均 Dice 替换）
weights = {
    "2d": 1.0,        # 你可以改成测试集 Dice
    "3d_lowres": 1.0,
    "3d_fullres": 1.0
}

output_dir = Path("data/nnunet/predictions/ensemble_failsafe_pred")
output_dir.mkdir(parents=True, exist_ok=True)

# 体素比例阈值（低于这个比例的模型将被忽略）
min_voxel_ratio = 0.00005  # 0.005% 的体素比例

# ===== Load case list from one model =====
case_files = sorted(Path(pred_dirs["2d"]).glob("*.nii.gz"))

print(f"🚀 Found {len(case_files)} cases for ensemble with fail-safe voting.")
for case_path in case_files:
    case_name = case_path.name
    preds = []
    used_models = []

    for model_name, model_dir in pred_dirs.items():
        model_file = Path(model_dir) / case_name
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found!")

        img = nib.load(str(model_file))
        data = img.get_fdata().astype(np.float32)  # probability or binary mask

        # 计算预测体素比例
        voxel_ratio = np.sum(data > 0.5) / data.size

        if voxel_ratio >= min_voxel_ratio:
            preds.append(weights[model_name] * data)
            used_models.append(model_name)
        else:
            print(f"⚠️ Skipping {model_name} for {case_name} (voxel ratio={voxel_ratio:.6f})")

    if preds:
        avg_pred = np.sum(preds, axis=0) / sum(weights[m] for m in used_models)
        final_mask = (avg_pred >= 0.5).astype(np.uint8)
    else:
        # 如果所有模型都被跳过，就用 2d 模型的结果
        print(f"❗ All models skipped for {case_name}, fallback to 2d prediction.")
        img = nib.load(str(Path(pred_dirs["2d"]) / case_name))
        final_mask = (img.get_fdata() >= 0.5).astype(np.uint8)

    # 保存结果
    out_img = nib.Nifti1Image(final_mask, img.affine, img.header)
    nib.save(out_img, str(output_dir / case_name))

print(f"✅ Fail-safe ensemble predictions saved to: {output_dir}")
