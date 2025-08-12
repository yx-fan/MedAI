import os
import nibabel as nib

IMAGES_DIR = "data/raw/images"
MASKS_DIR = "data/raw/masks"

def check_nifti_dir(directory):
    bad_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = nib.load(fpath)
                _ = img.get_fdata()
            except Exception as e:
                bad_files.append((fpath, str(e)))
    return bad_files

bad_images = check_nifti_dir(IMAGES_DIR)
bad_masks = check_nifti_dir(MASKS_DIR)

if not bad_images and not bad_masks:
    print("✅ All NIfTI files loaded successfully.")
else:
    print("❌ Found problematic files:")
    if bad_images:
        print("\n[Images with errors]")
        for fpath, reason in bad_images:
            print(f"{fpath} --> {reason}")
    if bad_masks:
        print("\n[Masks with errors]")
        for fpath, reason in bad_masks:
            print(f"{fpath} --> {reason}")
