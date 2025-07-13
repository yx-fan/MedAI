import os
from pathlib import Path
import subprocess

# Configurations
nnunet_predict_cmd = "nnUNetv2_predict"
images_dir = Path("data/nnunet/nnUNet_raw/Dataset201_MyTask/imagesTr")
output_dir = Path("data/nnunet/nnUNet_raw/Dataset201_MyTask/predicted_labels")
output_dir.mkdir(exist_ok=True)

task = "201"
config = "2d"
fold = "0"

# Invoke nnUNet prediction command
cmd = [
    nnunet_predict_cmd,
    "-i", str(images_dir),
    "-o", str(output_dir),
    "-d", task,
    "-c", config,
    "-f", fold,
    "-chk", "checkpoint_best.pth",
    "-device", "cpu",
    "-npp", "4",
    "-nps", "2"
]

print("Running nnUNet prediction...")
print(" ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Prediction finished! Masks saved to {output_dir}")
