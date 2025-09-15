import os
import csv
import torch
import matplotlib.pyplot as plt
import wandb

# ==============================
# Helper: GPU memory logger
# ==============================
def log_gpu(stage: str):
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated("cuda") / 1024**2
        print(f"[GPU] {stage} | Allocated: {mem:.2f} MB")


# ==============================
# Helper: WandB prediction visualization
# ==============================
def log_prediction(image, label, pred, epoch):
    mid = image.shape[-1] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image[0, 0, :, :, mid], cmap="gray"); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(label[0, 0, :, :, mid]); plt.title("GT")
    plt.subplot(1, 3, 3); plt.imshow(pred[0, 1, :, :, mid]); plt.title("Pred")
    wandb.log({"Predictions": wandb.Image(plt)}, step=epoch)
    plt.close()
