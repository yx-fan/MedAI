#!/bin/bash
set -e

# ===== Parameters =====
CONFIG=${1:-2d}         # default 2d / 3d_lowres / 3d_fullres
FOLD=${2:-0}            # default fold 0
CHECKPOINT=${3:-checkpoint_best.pth}  # default checkpoint

# ===== Paths =====
TASK=201
DATASET_DIR="data/nnunet/nnUNet_raw/Dataset201_MyTask"
LABELS_TS="${DATASET_DIR}/labelsTs"
PRED_DIR="data/nnunet/predictions/${CONFIG}_pred"
DJFILE="${DATASET_DIR}/dataset.json"
PFILE="data/nnunet/nnUNet_preprocessed/Dataset201_MyTask/nnUNetPlans.json"
OUTPUT_JSON="data/nnunet/evaluation/${CONFIG}_fold${FOLD}_evaluation.json"

# ===== Create output dir =====
mkdir -p "$(dirname "$OUTPUT_JSON")"

# ===== Output Information =====
echo "📊 Starting evaluation..."
echo "Task=$TASK, Config=$CONFIG, Fold=$FOLD"
echo "Reference labels: $LABELS_TS"
echo "Predictions dir : $PRED_DIR"
echo "Output JSON     : $OUTPUT_JSON"

# ===== Execute Evaluation =====
nnUNetv2_evaluate_folder \
    "$LABELS_TS" \
    "$PRED_DIR" \
    -djfile "$DJFILE" \
    -pfile "$PFILE" \
    -o "$OUTPUT_JSON"

echo "✅ Evaluation finished for Config=$CONFIG, Fold=$FOLD!"
echo "Results saved to $OUTPUT_JSON"
