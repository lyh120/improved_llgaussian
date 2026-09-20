#!/usr/bin/env bash
# Example single-scene v2 run. Override paths through environment variables.
set -euo pipefail

SCENE_NAME="${SCENE_NAME:-chair}"
DATA_ROOT="${DATA_ROOT:-./dataset/LLRS-sRGB}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
GPU="${GPU:-0}"

bash scripts/train.sh \
    --data "${DATA_ROOT}/${SCENE_NAME}" \
    --model "${OUTPUT_ROOT}/${SCENE_NAME}_explicit_v2" \
    --gpu "$GPU" --iterations 8000 --warmup --wandb
