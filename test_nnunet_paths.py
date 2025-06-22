import os
from pathlib import Path

raw = os.environ.get("nnUNet_raw")
print("nnUNet_raw:", raw)

task_dir = Path(raw) / "Task201_MyTask"
print("Task201_MyTask exists:", task_dir.exists())
print("imagesTr exists:", (task_dir / "imagesTr").exists())
print("labelsTr exists:", (task_dir / "labelsTr").exists())
print("dataset.json exists:", (task_dir / "dataset.json").exists())

print("Sample imagesTr:", list((task_dir / "imagesTr").glob("*.nii.gz"))[:3])
print("Sample labelsTr:", list((task_dir / "labelsTr").glob("*.nii.gz"))[:3])
