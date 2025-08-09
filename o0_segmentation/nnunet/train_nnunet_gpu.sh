#!/bin/bash
set -e

# Check if the script is called with the correct number of arguments
if [ $# -gt 3 ]; then
  echo "Usage: bash train_gpu.sh <CONFIG> [FOLD] [GPU_ID]"
  echo "Example: bash train_gpu.sh 3d_fullres 0 1"
  exit 1
fi

# Variables with defaults
CONFIG=${1:-2d}   # Must be 2d, 3d_lowres, 3d_fullres
FOLD=${2:-0}      # Fold number, default 0
GPU_ID=${3:-0}    # GPU number, default 0

# --- Validity checks ---
VALID_CONFIGS=("2d" "3d_lowres" "3d_fullres")
if [[ ! " ${VALID_CONFIGS[@]} " =~ " ${CONFIG} " ]]; then
  echo "[ERROR] Invalid CONFIG: ${CONFIG}"
  echo "Valid options are: ${VALID_CONFIGS[*]}"
  exit 1
fi

if ! [[ "$FOLD" =~ ^[0-4]$ ]]; then
  echo "[ERROR] Invalid FOLD: ${FOLD} (must be 0-4)"
  exit 1
fi

if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Invalid GPU_ID: ${GPU_ID} (must be integer)"
  exit 1
fi

# --- Set environment variables ---
export CUDA_VISIBLE_DEVICES=$GPU_ID
export nnUNet_raw="$(pwd)/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnunet/nnUNet_trained_models"

TASK=201

echo "🚀 Starting nnUNetv2 training..."
echo "Task=${TASK}, Config=${CONFIG}, Fold=${FOLD}, GPU=${GPU_ID}"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

nnUNetv2_train $TASK $CONFIG $FOLD -device cuda

echo "✅ Training finished for Config ${CONFIG}, Fold ${FOLD}!"
