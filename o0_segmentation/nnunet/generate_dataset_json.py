from pathlib import Path
import json

DATASET_DIR = Path("data/nnunet/nnUNet_raw/Dataset201_MyTask")
IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

images = sorted([f for f in IMAGES_TR.glob("*.nii.gz")])
labels = sorted([f for f in LABELS_TR.glob("*.nii.gz")])

assert len(images) == len(labels), "Number of images and labels must match."

dataset = {
    "name": "MyTask",
    "description": "My dataset for nnunet",
    "tensorImageSize": "3D",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "channel_names": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "tumor": 1
    },
    "numTraining": len(images),
    "numTest": 0,
    "training": [
        {
            "image": f"./imagesTr/{img.name}",
            "label": f"./labelsTr/{img.name.replace('_0000', '')}"
        }
        for img in images
    ],
    "test": [],
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}

DATASET_DIR.mkdir(parents=True, exist_ok=True)

print("Writing dataset.json to:", (DATASET_DIR / "dataset.json").resolve())
with open(DATASET_DIR / "dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("dataset.json generated!")
