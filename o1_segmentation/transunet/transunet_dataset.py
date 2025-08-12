import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class TransUNetDataset(Dataset):
    def __init__(self, img_dir, img_size=(128, 128), mode="npy", data_format="2d"):
        """
        适配格式:
        - npy 文件 shape = (N, 2, H, W)
          通道 0 = CT 图像, 通道 1 = mask
        - 支持 2D / 2.5D / 3D 数据
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        self.data_format = data_format.lower()
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])

        if not self.files:
            raise RuntimeError(f"No npy files found in {img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        data = np.load(path)  # shape: (N, 2, H, W)

        imgs = torch.from_numpy(data[:, 0, :, :]).float()  # [N, H, W]
        masks = torch.from_numpy(data[:, 1, :, :]).long()  # [N, H, W]

        if self.data_format == "2d":
            # 取中心切片
            center_idx = imgs.shape[0] // 2
            imgs = imgs[center_idx].unsqueeze(0)  # [1, H, W]
            masks = masks[center_idx]             # [H, W]

        elif self.data_format == "2.5d":
            # 多切片当作多通道输入
            masks = masks[masks.shape[0] // 2]     # 中心 mask
            imgs = imgs.contiguous()               # [C=N, H, W]

        elif self.data_format == "3d":
            # 保留深度维
            imgs = imgs.unsqueeze(0)  # [1, N, H, W]
            masks = masks.unsqueeze(0)

        else:
            raise ValueError(f"Unsupported data_format: {self.data_format}")

        # 如果需要 resize
        if imgs.shape[-2:] != self.img_size:
            imgs = F.interpolate(imgs.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False).squeeze(0)
            if masks.ndim == 2:
                masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode="nearest").squeeze(0).squeeze(0).long()
            else:
                masks = F.interpolate(masks.unsqueeze(0).float(), size=self.img_size, mode="nearest").squeeze(0).long()

        return imgs, masks
