#!/bin/bash
# Preprocess prediction set (without mask) to generate 3D data

IMAGES_DIR="data/raw/images"
OUTPUT_DIR="data/processed/predict_3d"

python ./o0_preprocess/run_preprocess.py \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode predict \
    --format 3d
