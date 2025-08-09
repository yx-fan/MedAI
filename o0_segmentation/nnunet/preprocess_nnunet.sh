#!/bin/bash
set -e

export nnUNet_raw="$(pwd)/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnunet/nnUNet_trained_models"

TASK=201

echo "Step 1: Preparing dataset..."
python o0_segmentation/nnunet/prepare_nnunet_data.py

echo "Step 2: Generating dataset.json..."
python o0_segmentation/nnunet/generate_dataset_json.py

for CONFIG in 2d 3d_lowres 3d_fullres
do
    echo "Step 3: Running nnUNetv2 preprocessing for Task${TASK}, Config ${CONFIG}..."
    nnUNetv2_plan_and_preprocess -d $TASK -c $CONFIG
done

echo "✅ All steps finished!"
