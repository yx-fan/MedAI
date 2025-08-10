#!/bin/bash
set -e

# ===== Parameters =====
CONFIG=${1:-2d}         # default 2d, 2d / 3d_lowres / 3d_fullres
FOLD=${2:-0}            # default fold 0
CHECKPOINT=${3:-checkpoint_best.pth}  # default checkpoint

# ===== Paths =====
TASK=201
LABELS_TS="data/nnunet/nnUNet_raw/Dataset201_MyTask/labelsTs"
PRED_DIR="data/nnunet/predictions/${CONFIG}_pred"

# ===== Output Information =====
echo "📊 Starting evaluation..."
echo "Task=$TASK, Config=$CONFIG, Fold=$FOLD"
echo "Reference labels: $LABELS_TS"
echo "Predictions dir : $PRED_DIR"

# ===== Execute Evaluation =====
nnUNetv2_evaluate_folder \
    -ref "$LABELS_TS" \
    -pred "$PRED_DIR" \
    -l 1 \
    -d "$TASK" \
    -c "$CONFIG"

echo "✅ Evaluation finished for Config=$CONFIG, Fold=$FOLD!"