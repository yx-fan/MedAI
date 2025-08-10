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

output_dir = Path("data/nnunet/predictions/ensemble_voting_pred")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== Case list (based on 2D predictions) =====
case_files = sorted(Path(pred_dirs["2d"]).glob("*.nii.gz"))

print(f"🗳 Found {len(case_files)} cases for voting ensemble.")
for case_path in case_files:
    case_name = case_path.name

    binary_preds = []
    for model_name, model_dir in pred_dirs.items():
        model_file = Path(model_dir) / case_name
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found!")

        img = nib.load(str(model_file))
        data = img.get_fdata().astype(np.float32)
        
        # Threshold to binary mask
        binary_mask = (data >= 0.5).astype(np.uint8)
        binary_preds.append(binary_mask)

    # Stack and sum across models
    summed = np.sum(binary_preds, axis=0)

    # Majority voting: 2 or more votes -> 1
    final_mask = (summed >= 2).astype(np.uint8)

    # Save final mask
    out_img = nib.Nifti1Image(final_mask, img.affine, img.header)
    nib.save(out_img, str(output_dir / case_name))

print(f"✅ Voting ensemble predictions saved to: {output_dir}")
