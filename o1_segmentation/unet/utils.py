import torch
import matplotlib.pyplot as plt


def log_gpu(stage: str):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated("cuda") / 1024**2
        print(f"[GPU] {stage} | Allocated: {mem:.2f} MB")


def log_prediction(writer, image, label, pred, epoch):
    mid = image.shape[-1] // 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image[0, 0, :, :, mid], cmap="gray")
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(label[0, 0, :, :, mid])
    ax[1].set_title("GT")
    ax[1].axis("off")

    ax[2].imshow(pred[0, 1, :, :, mid])
    ax[2].set_title("Pred")
    ax[2].axis("off")

    writer.add_figure("Predictions", fig, global_step=epoch)
    plt.close(fig)
