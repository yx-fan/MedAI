#!/bin/bash
# Preprocess prediction set (without mask) to generate 2.5D data

IMAGES_DIR="data/raw/images"
OUTPUT_DIR="data/processed/predict_2_5d"

STACK_N=5

python ./o0_preprocess/run_preprocess.py \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode predict \
    --format 2.5d \
    --N $STACK_N
