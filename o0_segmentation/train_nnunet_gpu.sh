#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export nnUNet_raw="$(pwd)/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnunet/nnUNet_trained_models"

TASK=201
CONFIG=2d
FOLD=0

echo "Starting nnUNetv2 training for Task${TASK}, Config ${CONFIG}, Fold ${FOLD} on GPU ${CUDA_VISIBLE_DEVICES}..."

nnUNetv2_train $TASK $CONFIG $FOLD -device cuda

echo "Training finished!"
