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
    encoder = get_resnet18_encoder(device=device)
    features, errors = extract_features(img_paths, encoder, device=device)

    np.save(out_feature_file, features)
    feat_df = pd.DataFrame(features, index=meta.index)
    feat_df.to_csv(out_feature_csv, index=False)
    print(f"Extracted features saved to {out_feature_file} and {out_feature_csv}")
    if errors:
        with open("feature_extraction_errors.txt", "w") as fout:
            fout.write('\n'.join(errors))
        print(f"{len(errors)} files failed, see feature_extraction_errors.txt")

if __name__ == "__main__":
    run_feature_extraction()
