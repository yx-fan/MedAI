#!/bin/bash
set -e

# ===== Parameters =====
CONFIG=${1:-2d}         # default 2d, 2d / 3d_lowres / 3d_fullres
FOLD=${2:-0}            # default fold 0
DEVICE=${3:-cuda}       # default use GPU, change to cpu to run on CPU
CHECKPOINT=${4:-checkpoint_best.pth}  # default best checkpoint

# ===== Paths =====
TASK=201
IMAGES_TS="data/nnunet/nnUNet_raw/Dataset201_MyTask/imagesTs"
OUTPUT_DIR="data/nnunet/predictions/${CONFIG}_pred"
mkdir -p "$OUTPUT_DIR"

# ===== Output Information =====
echo "🚀 Starting prediction..."
echo "Task=$TASK, Config=$CONFIG, Fold=$FOLD, Device=$DEVICE"
echo "Input images: $IMAGES_TS"
echo "Output dir  : $OUTPUT_DIR"

# ===== Execute Prediction =====
nnUNetv2_predict \
    -i "$IMAGES_TS" \
    -o "$OUTPUT_DIR" \
    -d "$TASK" \
    -c "$CONFIG" \
    -f "$FOLD" \
    -chk "$CHECKPOINT" \
    -device "$DEVICE"

echo "✅ Prediction finished for Config=$CONFIG, Fold=$FOLD!"
echo "Results saved to $OUTPUT_DIR"
