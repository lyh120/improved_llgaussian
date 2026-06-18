#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liuyuhao/ll_further/LL-Gaussian-sg}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/asg_cidnet_8k_batch4_v2}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --eval
  --gpu "${GPU_ID}"
  --use_asg_illumination
  --illumination_mode asg
  --asg_lobes 1
  --asg_lambda_min 1.0
  --asg_energy_reg 1e-4
  --asg_sharpness_reg 5e-5
  --asg_anisotropy_reg 1e-5
  --use_3D_filter
  --use_residual
  --use_wandb
  --warmup
  --iterations 8000
  --save_iterations 8000
  --test_iterations 4000 6000 8000
  --position_lr_max_steps 8000
  --offset_lr_max_steps 8000
  --voxel_size 0.0005
  --prune_ratio 1.0
  --start_stat 200
  --update_from 800
  --update_until 6000
  --update_interval 50
  --success_threshold 0.6
  --densify_grad_threshold 0.0001
  --min_opacity 0.002
  --mlp_opacity_lr_max_steps 8000
  --mlp_cov_lr_max_steps 8000
  --mlp_color_lr_max_steps 8000
  --mlp_color_lr_init 0.008
  --mlp_color_lr_final 0.00025
  --offset_lr_init 0.001
  --offset_lr_final 0.00001
  --feat_dim 32
  --reflectance_consistency_reg 2e-5
  --reflectance_smooth_reg 0.0
  --reflectance_edge_reg 2e-4
  --reflectance_edge_uplift_reg 3e-3
  --reflectance_contrast_reg 2e-3
  --reflectance_highfreq_reg 3e-3
  --highlight_reflectance_reg 1e-3
  --residual_chroma_reg 5e-4
  --reflectance_detail_reg 1e-6
  --reflectance_decoder_reg 2e-5
  --reflectance_offset_lr 0.008
  --reflectance_decoder_lr 0.002
  --b0_spatial_smooth_reg 0.0
  --residual_start_iter 3000
  --residual_ramp_iters 2500
  --enhancement_reflectance_reg 0.06
  --enhancement_degree_reg 0.2
  --enhancement_degree_global_reg 0.05
  --enhancement_smooth_reg 4e-4
  --enhancement_diff_start_iter 2350
  --enhancement_color_reg 0.06
  --enhancement_color_std_reg 0.02
  --enhancement_green_bias_reg 0.06
  --enhancement_prior cidnet
  --cidnet_conda_env CIDNet
  --cidnet_root ./submodules/HVI-CIDNet
  --cidnet_weights ./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth
  --cidnet_force_refresh
  --cidnet_refresh_interval 1500
  --cidnet_mlp_steps 100
  --cidnet_target_exposure 0.5
  --cidnet_refresh_reg 0.5
  --cidnet_color_reg 0.2
  --cidnet_param_reg 0.1
  --cidnet_mv_reg 0.5
)

run_scene() {
  local scene="$1"
  local output="$2"
  shift 2
  local log_file="${LOG_DIR}/${scene}_asg_cidnet_8k.log"

  echo "[$(date '+%F %T')] start ${scene}, log=${log_file}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" \
    -s "datasets/${scene}" \
    -m "${output}" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] finished ${scene}"
}

run_scene chair outputs/chair_cidnet_asg_enh_8k_batch4_v2
run_scene sofa outputs/sofa_cidnet_asg_enh_8k_batch4_v2 --cidnet_alpha_init 1.3 --cidnet_gamma_init 1.3
run_scene bike outputs/bike_cidnet_asg_enh_8k_batch4_v2 --cidnet_alpha_init 0.9 --cidnet_gamma_init 0.9
run_scene buu outputs/buu_cidnet_asg_enh_8k_batch4_v2 --cidnet_alpha_init 1.2 --cidnet_gamma_init 1.2

echo "[$(date '+%F %T')] all four ASG CIDNet 8k scenes finished"
