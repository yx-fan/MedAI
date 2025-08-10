import os
import nibabel as nib
import numpy as np
from pathlib import Path

# ===== Prediction folders =====
pred_dirs = {
    "2d": "data/nnunet/predictions/2d_pred",
    "3d_lowres": "data/nnunet/predictions/3d_lowres_pred",
    "3d_fullres": "data/nnunet/predictions/3d_fullres_pred"
}

output_dir = Path("data/nnunet/predictions/ensemble_pred")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== Weights (can be adjusted for non-uniform weighting) =====
weights = {
    "2d": 1.0,
    "3d_lowres": 1.0,
    "3d_fullres": 1.0
}

# ===== Get case file list (based on 2D predictions) =====
case_files = sorted(Path(pred_dirs["2d"]).glob("*.nii.gz"))

print(f"🚀 Found {len(case_files)} cases for ensemble.")
for case_path in case_files:
    case_name = case_path.name

    preds = []
    for model_name, model_dir in pred_dirs.items():
        model_file = Path(model_dir) / case_name
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found!")

        img = nib.load(str(model_file))
        data = img.get_fdata().astype(np.float32)  # 转成float防止溢出
        preds.append(weights[model_name] * data)

    # ===== Fusion (Weighted Average) =====
    avg_pred = np.sum(preds, axis=0) / sum(weights.values())

    # ===== Thresholding to get final mask =====
    final_mask = (avg_pred >= 0.5).astype(np.uint8)

    # ===== Save result =====
    out_img = nib.Nifti1Image(final_mask, img.affine, img.header)
    nib.save(out_img, str(output_dir / case_name))

print(f"✅ Ensemble predictions saved to: {output_dir}")
