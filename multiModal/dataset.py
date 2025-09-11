import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal tasks (CT image + clinical data).
    Always returns all labels: (time, event, ln_label), so multitask training is easy.
    """

    def __init__(self, meta_csv, clinical_csv, processed_dir="data/processed", split="train",
                 clinical_features=None, transform=None, agg="mean"):
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta["split"] == split].copy() if "split" in self.meta.columns else self.meta

        self.clinical = pd.read_csv(clinical_csv)

        # Merge meta with clinical by patient_id
        self.df = pd.merge(self.meta, self.clinical, on="patient_id", how="inner")

        # Clinical features
        # exclude_cols = ["patient_id", "time", "event", "ln_label"]
        exclude_cols = [
            "patient_id", "time", "event", "ln_label",
            # 🔥 防泄露：以下这些在训练时要去掉
            "lymph_node_metastasis",
            "number_of_metastatic_lymph_nodes_on_pathology",
            "N", "N_1", "N_2",
            "stage", "stage_standard", "stage_2", "stage_3", "stage_4",
            "stage_standard_2", "stage_standard_3", "stage_standard_4"
        ]
        if clinical_features is None:
            self.clinical_features = [c for c in self.clinical.columns if c not in exclude_cols]
        else:
            self.clinical_features = clinical_features

        self.processed_dir = processed_dir
        self.transform = transform
        self.agg = agg

        # Group by patient_id
        self.grouped = self.df.groupby("patient_id")
        self.patients = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        group = self.grouped.get_group(patient_id)

        # ----- Image -----
        images = []
        for _, row in group.iterrows():
            npy_path = os.path.join(self.processed_dir, row["npy_file"])
            arr = np.load(npy_path)  # shape: (2, N, H, W)
            if self.transform:
                arr = self.transform(arr)
            images.append(torch.tensor(arr, dtype=torch.float32))

        if self.agg == "mean":
            image_tensor = torch.stack(images, dim=0).mean(dim=0)
        elif self.agg == "max":
            image_tensor = torch.stack(images, dim=0).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

        # ----- Clinical -----
        clinical_vals = group.iloc[0][self.clinical_features].values.astype(np.float32)
        clinical_tensor = torch.tensor(clinical_vals, dtype=torch.float32)

        # ----- Labels -----
        time = torch.tensor(group.iloc[0]["time"], dtype=torch.float32)
        event = torch.tensor(group.iloc[0]["event"], dtype=torch.float32)
        ln_label = torch.tensor(group.iloc[0]["ln_label"], dtype=torch.long) if "ln_label" in group.columns else torch.tensor(-1)

        # Always return everything
        return image_tensor, clinical_tensor, time, event, ln_label, str(patient_id)
