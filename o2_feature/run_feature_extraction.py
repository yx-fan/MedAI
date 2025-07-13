import pandas as pd
import numpy as np
from pathlib import Path
from feature_extractor import get_resnet18_encoder, extract_features

def run_feature_extraction(
    meta_csv="data/processed/meta.csv",
    processed_dir="data/processed",
    out_feature_file="data/features/features.npy",
    out_feature_csv="data/features/features.csv",
    device='cpu'
):
    meta = pd.read_csv(meta_csv)
    img_paths = [Path(processed_dir) / f for f in meta['npy_file']]

    # Auto-detect channel number from the first sample
    if not img_paths:
        print("No input files found!")
        return
    arr = np.load(str(img_paths[0]))
    if arr.ndim == 4:
        n_input_channels = arr.shape[0] * arr.shape[1]  # e.g. (2, 5, 256, 256) -> 10
    elif arr.ndim == 3:
        n_input_channels = arr.shape[0]                 # e.g. (2, 256, 256) -> 2
    else:
        raise ValueError(f"Unknown npy shape: {arr.shape}")

    print(f"Auto-detected input channels: {n_input_channels}")

    encoder = get_resnet18_encoder(n_input_channels=n_input_channels, device=device)
    features, errors = extract_features(img_paths, encoder, device=device)  # shape: (num_samples, feature_dim)

    np.save(out_feature_file, features)

    # Combine features with metadata
    feat_df = meta[['patient_id', 'slice_idx', 'npy_file']].copy()
    feat_dim = features.shape[1]
    for i in range(feat_dim):
        feat_df[f'feat_{i}'] = features[:, i]

    feat_df.to_csv(out_feature_csv, index=False)
    print(f"Extracted features saved to {out_feature_file} and {out_feature_csv}")

    if errors:
        with open("feature_extraction_errors.txt", "w") as fout:
            fout.write('\n'.join(errors))
        print(f"{len(errors)} files failed, see feature_extraction_errors.txt")

if __name__ == "__main__":
    run_feature_extraction()
