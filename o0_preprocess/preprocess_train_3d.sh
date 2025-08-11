#!/bin/bash
# 预处理训练集（有 mask），生成 3D 数据

IMAGES_DIR="data/raw/images"
MASKS_DIR="data/raw/masks"
OUTPUT_DIR="data/processed/train_3d"

BG_RATIO=0.1

python ./o0_preprocess/run_preprocess.py \
    --images_dir "$IMAGES_DIR" \
    --masks_dir "$MASKS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode train \
    --format 3d \
    --background_ratio $BG_RATIO
