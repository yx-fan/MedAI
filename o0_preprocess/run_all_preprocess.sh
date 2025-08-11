#!/bin/bash
# ===============================
# 批量预处理 CT 数据（有 mask + 无 mask）
# 格式：2D / 2.5D / 3D 全部生成
# ===============================

# 原始数据路径
IMAGES_DIR="data/raw/images"
MASKS_DIR="data/raw/masks"

# 输出总目录
OUTPUT_BASE="data/processed"

# background_ratio 控制训练时保留的无肿瘤切片比例
BG_RATIO=0.1

# 2.5D 堆叠层数（必须是奇数）
STACK_N=5

# ---------- 有 mask（训练） ----------
echo "=== Processing TRAIN datasets (with mask) ==="

for fmt in 2d 2.5d 3d; do
    echo "--- Training format: $fmt ---"
    python ./o0_preprocess/run_preprocess.py \
        --images_dir "$IMAGES_DIR" \
        --masks_dir "$MASKS_DIR" \
        --output_dir "$OUTPUT_BASE/train_${fmt//./_}" \
        --mode train \
        --format "$fmt" \
        --N $STACK_N \
        --background_ratio $BG_RATIO
done


# ---------- 无 mask（预测） ----------
echo "=== Processing PREDICT datasets (no mask) ==="

for fmt in 2d 2.5d 3d; do
    echo "--- Predict format: $fmt ---"
    python ./o0_preprocess/run_preprocess.py \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$OUTPUT_BASE/predict_${fmt//./_}" \
        --mode predict \
        --format "$fmt" \
        --N $STACK_N
done

echo "✅ All preprocessing finished!"
