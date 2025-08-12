#!/bin/bash
set -e

# ===== Parameters =====
CONFIG=${1:-2d}         # 2d / 3d_lowres / 3d_fullres / ensemble
FOLD=${2:-0}            # fold index

# ===== Paths =====
TASK=201
DATASET_DIR="data/nnunet/nnUNet_raw/Dataset201_MyTask"
LABELS_TS="${DATASET_DIR}/labelsTs"

# 如果是 ensemble 模型，预测结果放在 ensemble_pred 文件夹
if [ "$CONFIG" == "ensemble" ]; then
    PRED_DIR="data/nnunet/predictions/ensemble_pred"
elif [ "$CONFIG" == "ensemble_voting" ]; then
    PRED_DIR="data/nnunet/predictions/ensemble_voting_pred"
elif [ "$CONFIG" == "ensemble_failsafe" ]; then
    PRED_DIR="data/nnunet/predictions/ensemble_failsafe_pred"
else
    PRED_DIR="data/nnunet/predictions/${CONFIG}_pred"
fi

DJFILE="${DATASET_DIR}/dataset.json"
PFILE="data/nnunet/nnUNet_preprocessed/Dataset201_MyTask/nnUNetPlans.json"
OUTPUT_JSON="data/nnunet/evaluation/${CONFIG}_fold${FOLD}_evaluation.json"

mkdir -p "$(dirname "$OUTPUT_JSON")"

# ===== Info =====
echo "📊 Starting evaluation..."
echo "Task=$TASK"
echo "Config=$CONFIG"
echo "Fold=$FOLD"
echo "Reference labels: $LABELS_TS"
echo "Predictions dir : $PRED_DIR"
echo "Output JSON     : $OUTPUT_JSON"

# ===== Run evaluation =====
nnUNetv2_evaluate_folder \
    "$LABELS_TS" \
    "$PRED_DIR" \
    -djfile "$DJFILE" \
    -pfile "$PFILE" \
    -o "$OUTPUT_JSON"

echo "✅ Evaluation finished for $CONFIG (fold $FOLD)!"
echo "Results saved to $OUTPUT_JSON"
