import os, numpy as np, nibabel as nib
from transunet_config import *

def dice_score(pred, target, eps=1e-6):
    inter = np.sum((pred > 0) & (target > 0))  # foreground Dice for binary
    denom = np.sum(pred > 0) + np.sum(target > 0)
    return (2*inter + eps) / (denom + eps)

scores = []
for fname in os.listdir("./predictions_transunet"):
    pred = nib.load(os.path.join("./predictions_transunet", fname)).get_fdata().astype(np.uint8)
    gt_name = fname if os.path.exists(os.path.join(LABELS_TS, fname)) else fname.replace("_0000.nii.gz", ".nii.gz")
    gt = nib.load(os.path.join(LABELS_TS, gt_name)).get_fdata().astype(np.uint8)
    scores.append(dice_score(pred, gt))

print(f"Mean foreground Dice: {np.mean(scores):.4f}")
