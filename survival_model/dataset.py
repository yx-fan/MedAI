import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal survival analysis (CT image + clinical data).
    - Loads preprocessed 2.5D CT slices (npy)
    - Joins with clinical variables
    - Returns (image, clinical_features, time, event)
    """

    def __init__(self, meta_csv, clinical_csv, processed_dir="data/processed", split="train",
                 clinical_features=None, transform=None, agg="mean"):
        """
        Args:
            meta_csv (str): Path to meta.csv (slice-level info).
            clinical_csv (str): Path to clinical.csv (patient-level info).
            processed_dir (str): Directory containing processed npy files.
            split (str): Dataset split ("train", "val", "test").
            clinical_features (list[str]): List of clinical features to use. If None, use all except id/time/event.
            transform (callable, optional): Transform applied on image.
            agg (str): How to aggregate multiple slices ("mean", "max"). Works per patient.
        """
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta["split"] == split].copy() if "split" in self.meta.columns else self.meta

        self.clinical = pd.read_csv(clinical_csv)

        # Merge meta with clinical by patient_id
        self.df = pd.merge(self.meta, self.clinical, on="patient_id", how="inner")

        # Clinical features
        exclude_cols = ["patient_id", "time", "event"]
        if clinical_features is None:
            self.clinical_features = [c for c in self.clinical.columns if c not in exclude_cols]
        else:
            self.clinical_features = clinical_features

        self.processed_dir = processed_dir
        self.transform = transform
        self.agg = agg

        # Build patient-level index (group slices)
        self.grouped = self.df.groupby("patient_id")

        self.patients = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        group = self.grouped.get_group(patient_id)

        images = []
        for _, row in group.iterrows():
            npy_path = os.path.join(self.processed_dir, row["npy_file"])
            arr = np.load(npy_path)  # shape: (2, N, H, W)
            if self.transform:
                arr = self.transform(arr)
            images.append(torch.tensor(arr, dtype=torch.float32))

        # Aggregate slices
        if self.agg == "mean":
            image_tensor = torch.stack(images, dim=0).mean(dim=0)
        elif self.agg == "max":
            image_tensor = torch.stack(images, dim=0).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

        # Clinical features
        clinical_vals = group.iloc[0][self.clinical_features].values.astype(np.float32)
        clinical_tensor = torch.tensor(clinical_vals, dtype=torch.float32)

        # Survival labels
        time = torch.tensor(group.iloc[0]["time"], dtype=torch.float32)
        event = torch.tensor(group.iloc[0]["event"], dtype=torch.float32)

        return image_tensor, clinical_tensor, time, event
