import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, TverskyLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose

# ==============================
# FocalTverskyLossCompat (newly added)
# ==============================
class FocalTverskyLossCompat(nn.Module):
    """
    Focal Tversky Loss for 3D medical image segmentation.
    Combines Tversky loss with a focal component to focus on hard examples.
    """
    def __init__(self, include_background=False, to_onehot_y=True, softmax=True,
                 alpha=0.7, beta=0.3, gamma=0.75, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            alpha=alpha,
            beta=beta,
            reduction=reduction,
        )

    def forward(self, pred, target):
        base = self.tversky(pred, target)
        return base ** self.gamma


# ==============================
# Model Definition
# ==============================
def build_model(device):
    model = UNet(
        spatial_dims=3,          # 3D segmentation
        in_channels=1,           # single-channel CT
        out_channels=2,          # foreground + background
        channels=(32, 64, 128, 256, 512),  # channels at each layer
        strides=(2, 2, 2, 2),    # downsampling at each layer
        num_res_units=2,         # number of residual units
    ).to(device)
    return model


# ==============================
# Loss function (current: DiceCELoss)
# ==============================
# loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0)
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

# --- Modified: combined loss (commented out, kept for later use) ---
# ce_weight = torch.tensor([0.2, 0.8], device=device)
# loss_dicece = DiceCELoss(
#     include_background=False,
#     to_onehot_y=True, softmax=True,
#     lambda_dice=0.7, lambda_ce=0.3,
#     weight=ce_weight
# )
# loss_ftv = FocalTverskyLossCompat(
#     include_background=False,
#     to_onehot_y=True, softmax=True,
#     alpha=0.7, beta=0.3, gamma=0.75
# )
# loss_boundary = HausdorffDTLoss(
#     include_background=False,
#     to_onehot_y=True, softmax=True,
#     alpha=2.0
# )
# def total_loss(pred, target):
#     return 0.5 * loss_dicece(pred, target) + 0.4 * loss_ftv(pred, target) + 0.1 * loss_boundary(pred, target)


# ==============================
# Post transforms
# ==============================
post_pred = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True),
])
post_label = AsDiscrete(to_onehot=2)


# ==============================
# Metrics
# ==============================
dice_metric = DiceMetric(include_background=False, reduction="none")
# hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95, directed=True)
precision_metric = ConfusionMatrixMetric(metric_name="precision", reduction="mean", include_background=False)
recall_metric = ConfusionMatrixMetric(metric_name="recall", reduction="mean", include_background=False)
# miou_metric = ConfusionMatrixMetric(metric_name="jaccard", reduction="mean", include_background=False)
specificity_metric = ConfusionMatrixMetric(metric_name="specificity", reduction="mean", include_background=False)
