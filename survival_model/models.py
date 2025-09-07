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
        # after merge: C = 2*in_slices
        self.out_dim = out_dim
        # A tiny CNN backbone
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
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Fusion + Cox head
# -----------------------------
class MultiModalCox(nn.Module):
    """
    Fuse image embedding and clinical embedding, output a single risk score (higher = higher risk).
    """
    def __init__(self, img_embed_dim: int, clin_embed_dim: int, hidden: int = 128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(img_embed_dim + clin_embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)  # risk score
        )

    def forward(self, z_img, z_clin):
        z = torch.cat([z_img, z_clin], dim=1)
        risk = self.fuse(z).squeeze(1)  # (B,)
        return risk

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
    # sort by time descending so risk sets are cumulative sums
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]

    # log cumulative hazard denominator: logsumexp over risk of risk set
    # Use a stable cumulative logsumexp
    # cum_logsumexp[i] = log(sum_{j<=i} exp(r[j]))
    max_so_far = torch.cumsum(torch.maximum(r, torch.tensor(0., device=r.device)), dim=0) * 0  # placeholder
    # Efficient cumlogsumexp
    # Convert to cumulative by using trick:
    # cum_logsumexp[i] = log(exp(r[0]) + ... + exp(r[i]))
    # We'll compute via running max for stability
    cum_max = torch.cummax(r, dim=0).values
    exp_shifted = torch.exp(r - cum_max)
    cum_exp = torch.cumsum(exp_shifted, dim=0)
    cum_logsumexp = cum_max + torch.log(cum_exp + 1e-12)

    # log-likelihood for events only
    loglik = r - cum_logsumexp
    neg_partial_ll = -(loglik * e).sum() / (e.sum() + 1e-8)
    return neg_partial_ll
