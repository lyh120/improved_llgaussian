#!/usr/bin/env bash
# Usage: bash scripts/validate_llrs_chair_sg.sh DATASET MODEL_PATH GPU
set -euo pipefail
data=${1:?Supply LLRS chair directory}
model=${2:?Supply output directory}
gpu=${3:-0}
export CUDA_VISIBLE_DEVICES="$gpu"
python train.py --eval -s "$data" -m "$model" --gpu -1 \
  --iterations 8000 --save_iterations 8000 --test_iterations 4000 8000 \
  --use_3D_filter --illumination_mode sg --sg_lobes 4 --reflectance_mode explicit \
  --enhancement_prior stablesr --voxel_size 0.001 --prune_ratio 0.1 \
  --supervision_profile llgaussian --use_residual \
  --offset_lr_init 0.001 --offset_lr_final 0.00001 \
  --mlp_color_lr_init 0.04 --mlp_color_lr_final 0.00025 \
  --pose_lr_init 0.0001 --pose_lr_final 0.00001 \
  --start_stat 500 --update_from 1000 --update_until 5000 --update_interval 100 \
  --position_lr_max_steps 8000 --offset_lr_max_steps 8000 \
  --mlp_opacity_lr_max_steps 8000 --mlp_cov_lr_max_steps 8000 \
  --mlp_color_lr_max_steps 8000 --mlp_featurebank_lr_max_steps 8000 \
  --appearance_lr_max_steps 8000 --pose_lr_max_steps 8000
mv "$model/test" "$model/test_training_entry_500"
python -c 'import sys; from utils.evaluation_utils import evaluate_saved_test_set; evaluate_saved_test_set(sys.argv[1], sys.argv[2])' "$model/test_training_entry_500/ours_8000" "$data"
python render.py -m "$model" --dataset_path "$data" --iteration 8000 --skip_train
mv "$model/test" "$model/test_optimized_50"
python render.py -m "$model" --dataset_path "$data" --iteration 8000 --skip_train --skip_optimize --profile_render_timing
