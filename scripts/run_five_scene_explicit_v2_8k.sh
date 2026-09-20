#!/usr/bin/env bash
# Sequential five-scene 8k v2 experiments with identical loss/appearance settings.
set -euo pipefail

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./experiments}"
SCENES=(buu sofa chair bike shrub)

for scene in "${SCENES[@]}"; do
    bash scripts/train.sh \
        --data "${DATA_ROOT}/${scene}" \
        --model "${OUTPUT_ROOT}/${scene}_explicit_v2_8k" \
        --gpu "$GPU" --iterations 8000 --warmup --wandb
done
