import os
import csv
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, EnsureType

def load_nifti_as_tensor(path):
    img = nib.load(path)
    data = img.get_fdata()
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
    return data

def evaluate(pred_dir, label_dir, out_csv="dice_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=False, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
    results = []

    for f in tqdm(pred_files, desc="Evaluating Dice"):
        pred_path = os.path.join(pred_dir, f)
        label_path = os.path.join(label_dir, f.replace("_pred", ""))

        if not os.path.exists(label_path):
            print(f"[WARN] Missing label for {f}, skipped.")
            continue

        pred = load_nifti_as_tensor(pred_path)
        label = load_nifti_as_tensor(label_path)

        pred, label = EnsureType()(pred), EnsureType()(label)
        pred, label = post_pred(pred).to(device), post_label(label).to(device)

        dice = dice_metric(y_pred=pred, y=label).item()
        results.append({"case": f, "dice": dice})

    avg_dice = np.mean([r["dice"] for r in results])
    print(f"\nAverage Dice = {avg_dice:.4f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice"])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({"case": "Average", "dice": avg_dice})

    print(f"[INFO] Results saved to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate segmentation results (Dice per case)")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted .nii.gz files")
    parser.add_argument("--label_dir", required=True, help="Directory containing ground-truth labels")
    parser.add_argument("--out_csv", default="dice_results.csv", help="Output CSV path")
    args = parser.parse_args()

    evaluate(args.pred_dir, args.label_dir, args.out_csv)
