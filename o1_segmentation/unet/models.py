import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, TverskyLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose


class FocalTverskyLossCompat(nn.Module):
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


def build_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


def build_loss_fn(device, use_combined=True):
    if not use_combined:
        return DiceCELoss(to_onehot_y=True, softmax=True)
    
    ce_weight = torch.tensor([0.1, 0.9], device=device)  # Adjusted for severe class imbalance (2262:1)
    loss_dicece = DiceCELoss(
        include_background=False,
        to_onehot_y=True, softmax=True,
        lambda_dice=0.7, lambda_ce=0.3,
        weight=ce_weight
    )
    loss_ftv = FocalTverskyLossCompat(
        include_background=False,
        to_onehot_y=True, softmax=True,
        alpha=0.7, beta=0.3, gamma=0.75
    )
    loss_boundary = HausdorffDTLoss(
        include_background=False,
        to_onehot_y=True, softmax=True,
        alpha=2.0,
        reduction="mean"
    )
    
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_dicece = loss_dicece
            self.loss_ftv = loss_ftv
            self.loss_boundary = loss_boundary
        
        def forward(self, pred, target):
            dicece_loss = self.loss_dicece(pred, target)
            ftv_loss = self.loss_ftv(pred, target)
            
            # HausdorffDTLoss is very slow for 3D, skip it to avoid memory/time issues
            # Can be enabled later if needed, but it significantly slows training
            try:
                boundary_loss = self.loss_boundary(pred, target)
            except (RuntimeError, MemoryError, Exception):
                # Skip Hausdorff loss if it fails (too slow or OOM)
                boundary_loss = torch.tensor(0.0, device=pred.device, requires_grad=False)
            
            # Adjust weights: 0.6 DiceCE + 0.4 FocalTversky (Hausdorff skipped)
            if boundary_loss.item() == 0.0:
                return 0.6 * dicece_loss + 0.4 * ftv_loss
            else:
                return 0.5 * dicece_loss + 0.4 * ftv_loss + 0.1 * boundary_loss
    
    return CombinedLoss()


post_pred = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True),
])
post_label = AsDiscrete(to_onehot=2)

dice_metric = DiceMetric(include_background=False, reduction="none")
precision_metric = ConfusionMatrixMetric(metric_name="precision", reduction="mean", include_background=False)
recall_metric = ConfusionMatrixMetric(metric_name="recall", reduction="mean", include_background=False)
specificity_metric = ConfusionMatrixMetric(metric_name="specificity", reduction="mean", include_background=False)
