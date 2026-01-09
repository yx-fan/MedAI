import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, TverskyLoss
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
    
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_dicece = loss_dicece
            self.loss_ftv = loss_ftv
        
        def forward(self, pred, target):
            # DiceCE + FocalTversky combination is sufficient for medical segmentation
            # HausdorffDTLoss removed due to computational cost (very slow for 3D)
            # Boundary quality can be improved via post-processing (KeepLargestConnectedComponent)
            return 0.6 * self.loss_dicece(pred, target) + 0.4 * self.loss_ftv(pred, target)
    
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
