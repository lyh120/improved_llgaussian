#!/usr/bin/env bash
# Train v2; missing scene-local priors are generated once before optimization.
set -euo pipefail

DATA=""
MODEL=""
ENHANCEMENT_PRIOR=""
DEPTH_PRIOR=""
ENHANCEMENT_BACKEND="cidnet"
STABLESR_INPUT_GAIN="15.0"
GPU="0"
ITERATIONS="8000"
WARMUP=false
WANDB=false

usage() {
    echo "Usage: bash scripts/train.sh -d DATA -m MODEL [options]"
    echo "Options: --enhancement-backend cidnet|stablesr --stablesr-input-gain X"
    echo "         --enhancement-prior DIR --depth-prior DIR --gpu N --iterations N --warmup --wandb"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data) DATA="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        --enhancement-prior) ENHANCEMENT_PRIOR="$2"; shift 2 ;;
        --depth-prior) DEPTH_PRIOR="$2"; shift 2 ;;
        --enhancement-backend) ENHANCEMENT_BACKEND="$2"; shift 2 ;;
        --stablesr-input-gain) STABLESR_INPUT_GAIN="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --warmup) WARMUP=true; shift ;;
        --wandb) WANDB=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$DATA" || -z "$MODEL" ]]; then
    usage
    exit 2
fi

ARGS=(
    --eval --gpu "$GPU" -s "$DATA" -m "$MODEL"
    --use_3D_filter
    --iterations "$ITERATIONS"
    --save_iterations "$((ITERATIONS / 4))" "$((ITERATIONS / 2))" "$((ITERATIONS * 3 / 4))" "$ITERATIONS"
    --test_iterations "$((ITERATIONS / 4))" "$((ITERATIONS / 2))" "$((ITERATIONS * 3 / 4))" "$ITERATIONS"
    --prune_ratio 1.0 --update_until "$((ITERATIONS * 3 / 4))"
    --start_stat 500 --update_from 2000 --update_interval 100
    --max_anchors 30000 --max_new_anchors_per_update 256
    --densify_level_caps 128,80,48 --max_pruned_anchors_per_update 0
    --warmup_iterations 2000 --warmup_start_stat 200
    --warmup_update_from 1200 --warmup_update_until 1900
    --warmup_update_interval 200 --warmup_max_new_anchors 128
    --warmup_level_caps 64,40,24
    --warmup_densify_grad_threshold 0.00025
    --warmup_success_threshold 0.8
    --position_lr_max_steps "$ITERATIONS" --offset_lr_max_steps "$ITERATIONS"
    --mlp_opacity_lr_max_steps "$ITERATIONS" --mlp_cov_lr_max_steps "$ITERATIONS"
    --mlp_featurebank_lr_max_steps "$ITERATIONS" --pose_lr_max_steps "$ITERATIONS"
    --explicit_appearance_lr_init 0.008
    --explicit_appearance_lr_final 0.00005
    --lambda_dssim 0.2 --lambda_scaling 0.01
    --lambda_reflectance_reconstruction 1.0 --lambda_illumination 1.0
    --lambda_enhanced 1.0 --lambda_depth 1.0
    --enhancement_prior_backend "$ENHANCEMENT_BACKEND"
    --stablesr_input_gain "$STABLESR_INPUT_GAIN"
)
[[ -n "$ENHANCEMENT_PRIOR" ]] && ARGS+=(--enhancement_prior_path "$ENHANCEMENT_PRIOR")
[[ -n "$DEPTH_PRIOR" ]] && ARGS+=(--depth_prior_path "$DEPTH_PRIOR")
[[ "$WARMUP" == true ]] && ARGS+=(--warmup)
[[ "$WANDB" == true ]] && ARGS+=(--use_wandb)

python train.py "${ARGS[@]}"
