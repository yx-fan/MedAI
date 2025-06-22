#!/bin/bash

export nnUNet_raw="$(pwd)/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnunet/nnUNet_trained_models"

TASK=201
CONFIG=2d  # 这里改为2d

echo "Running nnUNetv2 preprocessing for Task${TASK}, Config ${CONFIG}..."

nnUNetv2_plan_and_preprocess -d $TASK -c $CONFIG

echo "Preprocessing finished!"