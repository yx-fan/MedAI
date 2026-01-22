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
    
    # 更激进的权重处理极度不平衡（2262:1）
    # 进一步增加前景权重，让模型更关注前景
    ce_weight = torch.tensor([0.02, 0.98], device=device)
    loss_dicece = DiceCELoss(
        include_background=False,
        to_onehot_y=True, softmax=True,
        lambda_dice=0.85, lambda_ce=0.15,  # 进一步增加Dice权重
        weight=ce_weight
    )
    # 调整FocalTversky参数，更关注假阴性（漏检）和难样本
    loss_ftv = FocalTverskyLossCompat(
        include_background=False,
        to_onehot_y=True, softmax=True,
        alpha=0.2, beta=0.8, gamma=1.5  # 增加gamma以更关注难样本
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
            # 调整权重：更依赖DiceCE（对不平衡数据更稳定）
            return 0.75 * self.loss_dicece(pred, target) + 0.25 * self.loss_ftv(pred, target)
    
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
