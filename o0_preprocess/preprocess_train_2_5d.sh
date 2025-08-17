#!/bin/bash
# Preprocess training set (with mask) to generate 2.5D data

IMAGES_DIR="data/raw/images"
MASKS_DIR="data/raw/masks"
OUTPUT_DIR="data/processed/train_2_5d"

BG_RATIO=0.1
STACK_N=5

python ./o0_preprocess/run_preprocess.py \
    --images_dir "$IMAGES_DIR" \
    --masks_dir "$MASKS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode train \
    --format 2.5d \
    --N $STACK_N \
    --background_ratio $BG_RATIO
