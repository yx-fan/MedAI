import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Small 2D CNN encoder for 2.5D inputs
# -----------------------------
class ImageEncoder2_5D(nn.Module):
    """
    Expect input shape: (B, 2, N, H, W)
    We merge (2, N) -> C channels (C = 2*N) and run a 2D CNN over (H, W).
    """
    def __init__(self, in_slices: int, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_slices, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global pool
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        # x: (B, 2, N, H, W)
        b, c2, n, h, w = x.shape
        assert c2 == 2, "channel 0=CT, channel 1=Mask expected"
        x = x.reshape(b, c2 * n, h, w)        # -> (B, 2N, H, W)
        x = self.conv(x)                      # -> (B, 256, 1, 1)
        x = x.flatten(1)                      # -> (B, 256)
        x = self.fc(x)                        # -> (B, out_dim)
        return x


# -----------------------------
# MLP for clinical features
# -----------------------------
class ClinicalMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Fusion backbone
# -----------------------------
class MultiModalBackbone(nn.Module):
    def __init__(self, img_embed_dim: int, clin_embed_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(img_embed_dim + clin_embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.hidden_dim = hidden

    def forward(self, z_img, z_clin):
        z = torch.cat([z_img, z_clin], dim=1)
        return self.fuse(z)


# -----------------------------
# Cox head (for survival risk)
# -----------------------------
class CoxHead(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, z):
        return self.fc(z).squeeze(1)  # risk score (B,)


# -----------------------------
# Classification head (for LN metastasis)
# -----------------------------
class ClassificationHead(nn.Module):
    def __init__(self, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, z):
        return self.fc(z)  # logits (B, num_classes)


# -----------------------------
# Combined MultiTask Model
# -----------------------------
class MultiModalNet(nn.Module):
    """
    Multi-task model:
      - Cox survival risk prediction
      - Lymph node metastasis classification
    """
    def __init__(self, img_embed_dim: int, clin_embed_dim: int, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.backbone = MultiModalBackbone(img_embed_dim, clin_embed_dim, hidden)
        self.cox_head = CoxHead(hidden)
        self.cls_head = ClassificationHead(hidden, num_classes)

    def forward(self, z_img, z_clin, task="survival"):
        z = self.backbone(z_img, z_clin)
        if task == "survival":
            return self.cox_head(z)  # risk score
        elif task == "ln_classification":
            return self.cls_head(z)  # logits
        else:
            raise ValueError(f"Unknown task: {task}")


# -----------------------------
# Cox partial likelihood loss
# -----------------------------
def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    risk: (B,) higher means higher hazard
    time: (B,) survival time
    event: (B,) 1=event, 0=censored
    Implements negative partial log-likelihood.
    """
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]

    # stable cumulative logsumexp
    cum_max = torch.cummax(r, dim=0).values
    exp_shifted = torch.exp(r - cum_max)
    cum_exp = torch.cumsum(exp_shifted, dim=0)
    cum_logsumexp = cum_max + torch.log(cum_exp + 1e-12)

    loglik = r - cum_logsumexp
    neg_partial_ll = -(loglik * e).sum() / (e.sum() + 1e-8)
    return neg_partial_ll
