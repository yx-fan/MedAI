import os
import pandas as pd
import numpy as np

def prepare_split(img_dir, train_ratio=0.8, seed=42, force=False):
    """
    Prepare train/test split at patient level.
    Ensures all slices from the same patient go into the same split.

    Args:
        img_dir (str): directory containing meta.csv
        train_ratio (float): proportion of patients assigned to train
        seed (int): random seed for reproducibility
        force (bool): if True, overwrite existing split column
    """
    meta_path = os.path.join(img_dir, "meta.csv")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"meta.csv not found in {img_dir}")

    meta = pd.read_csv(meta_path)

    if "patient_id" not in meta.columns:
        raise RuntimeError("meta.csv must have a 'patient_id' column for patient-level split.")

    if "split" in meta.columns and not force:
        print("⚠️ 'split' column already exists in meta.csv, skipping split generation.")
        return

    # unique patients
    patients = meta["patient_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(patients)

    # train/test split
    n_train = int(len(patients) * train_ratio)
    train_patients = set(patients[:n_train])
    test_patients = set(patients[n_train:])

    # assign split
    meta["split"] = meta["patient_id"].apply(lambda pid: "train" if pid in train_patients else "test")

    # save back
    meta.to_csv(meta_path, index=False)
    print(f"✅ meta.csv updated with patient-level split: "
          f"{len(train_patients)} train patients, {len(test_patients)} test patients")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare patient-level train/test split for meta.csv")
    parser.add_argument("img_dir", type=str, help="Directory containing meta.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of patients in train set (default=0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing split column")
    args = parser.parse_args()

    prepare_split(args.img_dir, args.train_ratio, args.seed, force=args.force)
