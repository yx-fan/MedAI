import os
from pathlib import Path
import numpy as np
import pandas as pd

from image_preprocess import match_ct_and_mask, get_processed_all_tumor_slices

def generate_processed_data(
    images_dir="data/raw/images/",
    masks_dir="data/raw/masks/",
    processed_data_dir="data/processed/"
):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    processed_data_dir = Path(processed_data_dir)
    os.makedirs(processed_data_dir, exist_ok=True)

    meta_info = []

    # Loop through all CT files in the images directory
    for ct_file in images_dir.glob("*.nii.gz"):
        patient_id = ct_file.name.replace('.nii.gz', '')
        mask_file = masks_dir / f"{patient_id}.nii.gz"
        print(f"Processing {patient_id}...")
        print(f"CT file: {ct_file}, Mask file: {mask_file}")
        if not mask_file.exists():
            print(f"[WARNING] Mask not found for {patient_id}. Skipping.")
            continue

        try:
            ct, mask, tumor_slices = match_ct_and_mask(str(ct_file), str(mask_file))
            print(f"Found {len(tumor_slices)} tumor slices for {patient_id}.")
        except Exception as e:
            print(f"[ERROR] {ct_file}, {mask_file}: {e}")
            continue

        slice_results = get_processed_all_tumor_slices(ct, mask, margin=10, out_size=(256,256))
        if len(slice_results) == 0:
            print(f"[INFO] No tumor slices for {patient_id}, skipping.")
            continue

        for processed, slice_idx in slice_results:
            out_name = f"{patient_id}_slice{slice_idx}.npy"
            np.save(processed_data_dir / out_name, processed)
            meta_info.append({
                "patient_id": patient_id,
                "slice_idx": slice_idx,
                "ct_file": os.path.relpath(ct_file, start=images_dir),
                "mask_file": os.path.relpath(mask_file, start=masks_dir),
                "npy_file": out_name
            })

    # Save metadata to CSV
    df_meta = pd.DataFrame(meta_info)
    if len(df_meta) > 0:
        df_meta = df_meta.sort_values(["patient_id", "slice_idx"])
        df_meta.to_csv(processed_data_dir / "meta.csv", index=False)
        print(f"Done! Processed {len(meta_info)} slices.")
    else:
        print("No valid tumor slices processed. Please check data!")