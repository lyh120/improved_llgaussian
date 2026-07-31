#!/usr/bin/env bash
# Render the LLNeRF still2--still4 test splits and compare enhanced renders to GT.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/liuyuhao/datasets/llnerf-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuyuhao/outputs/llgaussian_llnerf}"
GPU="${GPU:-1}"
ITERATION="${ITERATION:--1}"
SCENES=(still2 still3 still4)

cd "$PROJECT_ROOT"

for scene in "${SCENES[@]}"; do
    scene_data="$DATA_ROOT/$scene"
    model_path="$OUTPUT_ROOT/$scene"

    [[ -d "$scene_data" ]] || { echo "Missing scene data: $scene_data" >&2; exit 1; }
    [[ -d "$scene_data/gt" ]] || { echo "Missing evaluation GT: $scene_data/gt" >&2; exit 1; }
    [[ -d "$model_path" ]] || { echo "Missing trained model: $model_path" >&2; exit 1; }

    CUDA_VISIBLE_DEVICES="$GPU" python render.py \
        -m "$model_path" \
        --dataset_path "$scene_data" \
        --iteration "$ITERATION" \
        --skip_train \
        --skip_optimize
done
