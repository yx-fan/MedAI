import os
import csv
import torch
import matplotlib.pyplot as plt

# ==============================
# Helper: GPU memory logger
# ==============================
def log_gpu(stage: str):
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated("cuda") / 1024**2
        print(f"[GPU] {stage} | Allocated: {mem:.2f} MB")


# ==============================
# Helper: TensorBoard prediction visualization
# ==============================
def log_prediction(writer, image, label, pred, epoch):
    """
    Logs a mid-slice visualization (image, GT, Pred) to TensorBoard.
    Args:
        writer: TensorBoard SummaryWriter
        image: input image tensor (numpy) [B, C, H, W, D]
        label: ground truth mask (numpy) [B, C, H, W, D]
        pred: model prediction (numpy) [B, C, H, W, D]
        epoch: current epoch number
    """
    mid = image.shape[-1] // 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image[0, 0, :, :, mid], cmap="gray")
    ax[0].set_title("Image"); ax[0].axis("off")

    ax[1].imshow(label[0, 0, :, :, mid])
    ax[1].set_title("GT"); ax[1].axis("off")

    ax[2].imshow(pred[0, 1, :, :, mid])
    ax[2].set_title("Pred"); ax[2].axis("off")

    writer.add_figure("Predictions", fig, global_step=epoch)
    plt.close(fig)
