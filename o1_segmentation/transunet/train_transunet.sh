#!/bin/bash
# =========================================
#  TransUNet GPU Training Script
# =========================================

set -e  # 遇到错误直接退出
set -u  # 未定义变量时报错

# 1. 切换到脚本所在目录
cd "$(dirname "$0")"

# 2. 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found! Please activate your environment first."
    exit 1
fi

# 3. 检查 PyTorch GPU
python - << 'EOF'
import torch
if not torch.cuda.is_available():
    print("[WARNING] CUDA not available. Training will run on CPU.")
else:
    print(f"[INFO] CUDA available. Device count: {torch.cuda.device_count()} | Using: {torch.cuda.get_device_name(0)}")
EOF

# 4. 创建模型保存目录（路径从 config 里来）
MODEL_DIR="./checkpoints_transunet"
mkdir -p "$MODEL_DIR"

# 5. 启动训练
echo "[INFO] Starting TransUNet training..."
python train_transunet.py

# 6. 完成提示
echo "[INFO] Training completed. Models saved in $MODEL_DIR"
