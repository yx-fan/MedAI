import os
import torch

# ================== Data paths ==================
# Base directory where npy files are stored
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))

# Training / testing directories
IMAGES_TR = os.path.join(BASE_DIR, "train_2_5d")   # training npy files
IMAGES_TS = os.path.join(BASE_DIR, "test_2_5d")    # testing npy files

# Directory to save model checkpoints (auto-created if not exists)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "data/transunet/checkpoints_transunet")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ================== Training configuration ==================
IMG_SIZE = (256, 256)   # input image size (H, W)
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # auto switch

# ================== Model configuration ==================
VIT_PATCH_SIZE = 16     # patch size for ViT encoder
VIT_EMBED_DIM = 768     # embedding dimension
VIT_DEPTH = 12          # number of Transformer encoder layers
VIT_HEADS = 12          # number of attention heads
NUM_CLASSES = 2         # number of segmentation classes (including background)

# ================== Data format ==================
# "2d":   single-slice input  -> shape (1, H, W)
# "2.5d": multi-slice input   -> shape (N, H, W)
# "3d":   not supported by current TransUNet implementation
DATA_FORMAT = "2.5d"    # options: "2d" or "2.5d"
