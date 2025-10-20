import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose


# ============================================================
# 3D TransUNet Model (Enhanced Decoder)
# ============================================================
class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block with BN and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class PatchEmbedding3D(nn.Module):
    """Patch embedding for 3D input using Conv3D projection."""
    def __init__(self, in_ch=1, patch_size=8, emb_size=768, img_size=(128, 128, 64)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_ch, emb_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, emb_size))

    def forward(self, x):
        x = self.proj(x)  # [B, E, D', H', W']
        B, E, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        x = x + self.pos_emb
        return x, (D, H, W)


class TransformerEncoder3D(nn.Module):
    """Transformer encoder for 3D tokens."""
    def __init__(self, emb_size=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size, nhead=heads,
                dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransUNet3D(nn.Module):
    """3D TransUNet: ViT encoder + Enhanced CNN decoder"""
    def __init__(self,
                 in_ch=1,
                 out_ch=2,
                 img_size=(128, 128, 64),
                 patch_size=8,
                 emb_size=512,
                 depth=8,
                 heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_ch, patch_size, emb_size, img_size)
        self.transformer = TransformerEncoder3D(emb_size, depth, heads)

        # Decoder: enhanced, multi-stage upsampling
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(emb_size, 256, kernel_size=2, stride=2),
                ConvBlock3D(256, 256)
            ),
            nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
                ConvBlock3D(128, 128)
            ),
            nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
                ConvBlock3D(64, 64)
            ),
            nn.Sequential(
                nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
                ConvBlock3D(32, 32)
            )
        ])

        self.final_conv = nn.Conv3d(32, out_ch, kernel_size=1)

    def forward(self, x):
        B = x.size(0)
        x, (D, H, W) = self.patch_embed(x)
        x = self.transformer(x)
        x = x.transpose(1, 2).contiguous().view(B, -1, D, H, W)

        for block in self.decoder:
            x = block(x)
        x = self.final_conv(x)
        return x


# ============================================================
# Build + Loss + Metrics + Post-processing
# ============================================================
def build_model(device):
    """
    Create the 3D TransUNet model and move to device.
    """
    model = TransUNet3D(
        in_ch=1,
        out_ch=2,
        img_size=(128, 128, 64),
        patch_size=8,
        emb_size=512,
        depth=8,
        heads=8
    ).to(device)
    return model


# === Loss function ===
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

# === Post-processing ===
post_pred = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True),
])
post_label = AsDiscrete(to_onehot=2)

# === Metrics ===
dice_metric = DiceMetric(include_background=False, reduction="none")
precision_metric = ConfusionMatrixMetric(metric_name="precision", reduction="mean", include_background=False)
recall_metric = ConfusionMatrixMetric(metric_name="recall", reduction="mean", include_background=False)
specificity_metric = ConfusionMatrixMetric(metric_name="specificity", reduction="mean", include_background=False)
