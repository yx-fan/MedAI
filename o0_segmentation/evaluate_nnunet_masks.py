import os
from pathlib import Path
import numpy as np
import nibabel as nib

# Configurations
LABELS_TR = Path("data/nnunet/nnUNet_raw/Dataset201_MyTask/labelsTr")
PRED_LABELS = Path("data/nnunet/nnUNet_raw/Dataset201_MyTask/predicted_labels")

def dice_coefficient(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    pred_files = sorted(PRED_LABELS.glob("*.nii.gz"))
    if not pred_files:
        print(f"No predictions found in {PRED_LABELS}")
        return
    dice_scores = []
    for pred_file in pred_files:
        case_name = pred_file.name.replace(".nii.gz", "")
        gt_file = LABELS_TR / f"{case_name}.nii.gz"
        if not gt_file.exists():
            print(f"[WARNING] Ground truth for {case_name} not found, skipping.")
            continue
        pred = nib.load(str(pred_file)).get_fdata()
        gt = nib.load(str(gt_file)).get_fdata()
        dice = dice_coefficient(pred, gt)
        dice_scores.append(dice)
        print(f"Case: {case_name:12s} | Dice: {dice:.4f}")
    if dice_scores:
        print(f"\nAverage Dice: {np.mean(dice_scores):.4f} (n={len(dice_scores)})")
    else:
        print("No valid cases were evaluated.")

if __name__ == "__main__":
    main()
