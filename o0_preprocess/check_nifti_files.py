import os
import nibabel as nib

IMAGES_DIR = "data/raw/images"
MASKS_DIR = "data/raw/masks"

def check_nifti_pair(img_path, mask_path):
    try:
        img = nib.load(img_path).get_fdata()
    except Exception as e:
        return f"❌ Image load error: {e}"
    try:
        mask = nib.load(mask_path).get_fdata()
    except Exception as e:
        return f"❌ Mask load error: {e}"

    if img.shape != mask.shape:
        return f"⚠️ Shape mismatch: image {img.shape}, mask {mask.shape}"
    return None

bad_files = []
for root, _, files in os.walk(IMAGES_DIR):
    for fname in files:
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        pid = fname.replace(".nii.gz", "").replace(".nii", "")
        img_path = os.path.join(root, fname)
        mask_path = os.path.join(MASKS_DIR, fname)
        print(f"Checking {img_path} and {mask_path}")
        if not os.path.exists(mask_path):
            bad_files.append((img_path, "❌ Missing corresponding mask"))
            continue

        result = check_nifti_pair(img_path, mask_path)
        if result:
            bad_files.append((img_path, result))

if not bad_files:
    print("✅ All image-mask pairs are consistent and readable.")
else:
    print("❌ Found problematic pairs:")
    for fpath, reason in bad_files:
        print(f"{fpath} --> {reason}")
