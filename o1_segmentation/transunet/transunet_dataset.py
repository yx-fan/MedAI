import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class TransUNetDataset(Dataset):
    def __init__(self, img_dir, img_size=(256, 256), mode="npy", data_format="2.5d", split="train"):
        """
        Dataset for TransUNet training.
        Each .npy file has shape (2, N, H, W):
            - data[0] = CT stack (N, H, W)
            - data[1] = mask stack (N, H, W)

        Args:
            img_dir (str): directory containing .npy files
            img_size (tuple): resize target size (H, W)
            data_format (str): "2d" | "2.5d" | "3d"
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode.lower()
        self.data_format = data_format.lower()
        self.split = split.lower()

        # Load meta.csv
        meta_path = os.path.join(img_dir, "meta.csv")
        if not os.path.exists(meta_path):
            raise RuntimeError(f"meta.csv not found in {img_dir}")
        meta = pd.read_csv(meta_path)

        # Filter by split
        if "split" not in meta.columns:
            raise RuntimeError("meta.csv must have a 'split' column with values 'train' or 'test'")
        meta = meta[meta["split"] == self.split]

        if "npy_file" not in meta.columns:
            raise RuntimeError("meta.csv must have a 'npy_file' column containing .npy filenames")

        self.files = meta["npy_file"].tolist()

        if not self.files:
            raise RuntimeError(f"No npy files found in {img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        data = np.load(path)  # shape: (2, N, H, W)

        # split CT and mask
        imgs = torch.from_numpy(data[0]).float()  # [N, H, W]
        masks = torch.from_numpy(data[1]).long()  # [N, H, W]

        if self.data_format == "2d":
            # take the center slice (single channel)
            center_idx = imgs.shape[0] // 2
            imgs = imgs[center_idx].unsqueeze(0)  # [1, H, W]
            masks = masks[center_idx]             # [H, W]

        elif self.data_format == "2.5d":
            # use multi-slice as multi-channel input
            center_idx = masks.shape[0] // 2
            masks = masks[center_idx]             # [H, W], only center mask
            imgs = imgs.contiguous()              # [N, H, W], input channels = N

        elif self.data_format == "3d":
            # keep depth dimension for Conv3d models
            imgs = imgs.unsqueeze(0)  # [1, N, H, W]
            masks = masks.unsqueeze(0)

        else:
            raise ValueError(f"Unsupported data_format: {self.data_format}")

        # resize if needed
        if imgs.shape[-2:] != self.img_size:
            imgs = F.interpolate(
                imgs.unsqueeze(0), size=self.img_size,
                mode="bilinear", align_corners=False
            ).squeeze(0)

            if masks.ndim == 2:
                masks = F.interpolate(
                    masks.unsqueeze(0).unsqueeze(0).float(),
                    size=self.img_size, mode="nearest"
                ).squeeze(0).squeeze(0).long()
            else:
                masks = F.interpolate(
                    masks.unsqueeze(0).float(),
                    size=self.img_size, mode="nearest"
                ).squeeze(0).long()

        return imgs, masks
