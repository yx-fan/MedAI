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
    # 降低模型容量以减少过拟合：从(32,64,128,256,512)降到(24,48,96,192,384)
    # 减少约30%参数量，提升泛化能力
    # 注意：MONAI UNet可能不支持dropout参数，通过降低通道数和weight_decay来正则化
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(24, 48, 96, 192, 384),  # 减少通道数，降低模型容量
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


def build_loss_fn(device, use_combined=True):
    if not use_combined:
        return DiceCELoss(to_onehot_y=True, softmax=True)
    
    # 处理极度不平衡（2262:1）
    # CE weight: 背景:前景 = 1:50，归一化后约为 [0.02, 0.98]
    # 使用更直观的比例表示，但保持归一化（PyTorch CrossEntropyLoss需要）
    ce_weight = torch.tensor([1.0, 50.0], device=device)
    ce_weight = ce_weight / ce_weight.sum()  # 归一化到 [0.0196, 0.9804]
    
    loss_dicece = DiceCELoss(
        include_background=False,
        to_onehot_y=True, softmax=True,
        lambda_dice=0.7, lambda_ce=0.3,  # 增加CE权重以更好惩罚假阳性
        weight=ce_weight
    )
    # FocalTversky参数：alpha > beta 以减少假阳性（提升Precision）
    # alpha控制假阳性(FP)惩罚，beta控制假阴性(FN)惩罚
    # 当前Precision低(0.42)，需要更严厉惩罚假阳性
    loss_ftv = FocalTverskyLossCompat(
        include_background=False,
        to_onehot_y=True, softmax=True,
        alpha=0.7, beta=0.3, gamma=1.5  # alpha > beta 减少假阳性
    )
    
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_dicece = loss_dicece
            self.loss_ftv = loss_ftv
        
        def forward(self, pred, target):
            # 调整权重：更依赖DiceCE以提升稳定性，减少过拟合风险
            # DiceCE对不平衡数据更稳定，FocalTversky关注难样本但可能增加不稳定性
            return 0.7 * self.loss_dicece(pred, target) + 0.3 * self.loss_ftv(pred, target)
    
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
