import os
import torch

# ================== 数据路径 ==================
# npy 文件所在的目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))

# 训练集 / 测试集目录
IMAGES_TR = os.path.join(BASE_DIR, "train_2_5d")   # 存放训练 npy
IMAGES_TS = os.path.join(BASE_DIR, "test_2_5d")    # 存放测试 npy

# 模型保存目录（自动创建）
MODEL_SAVE_DIR = os.path.abspath("./checkpoints_transunet")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ================== 训练配置 ==================
IMG_SIZE = (128, 128)   # 输入图像大小（H, W）
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动切换

# ================== 模型配置 ==================
VIT_PATCH_SIZE = 16
VIT_EMBED_DIM = 768
VIT_DEPTH = 12
VIT_HEADS = 12
NUM_CLASSES = 2  # 包含背景类

# ================== 数据格式设置 ==================
# 2D 模式：单通道输入 (1, H, W)
# 2.5D 模式：多切片输入 (N, H, W)
# 3D 模式：当前 TransUNet 不直接支持
DATA_FORMAT = "2.5d"   # 可选 "2d" / "2.5d"
