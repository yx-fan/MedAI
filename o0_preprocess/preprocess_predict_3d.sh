#!/bin/bash
# 预处理预测集（无 mask），生成 3D 数据

IMAGES_DIR="data/raw/images"
OUTPUT_DIR="data/processed/predict_3d"

python ./o0_preprocess/run_preprocess.py \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode predict \
    --format 3d
