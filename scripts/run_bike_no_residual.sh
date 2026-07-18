#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
OUTPUT="outputs/bike_ablation_no_residual_8k"
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')

if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing experiment: $OUTPUT" >&2
    exit 1
fi

echo "Training bike no-residual ablation on GPU $GPU, port $PORT"
echo "CIDNet policy: fixed round_000 (--cidnet_refresh_interval 0, no force refresh)"

python train.py \
    -s datasets/bike \
    -m "$OUTPUT" \
    --port "$PORT" \
    --eval \
    --gpu "$GPU" \
    --use_asg_illumination \
    --illumination_mode asg \
    --asg_lobes 1 \
    --asg_lambda_min 1.0 \
    --asg_energy_reg 1e-4 \
    --asg_sharpness_reg 5e-5 \
    --asg_anisotropy_reg 1e-5 \
    --use_3D_filter \
    --use_wandb \
    --warmup \
    --iterations 8000 \
    --save_iterations 8000 \
    --test_iterations 4000 6000 8000 \
    --position_lr_max_steps 8000 \
    --offset_lr_max_steps 8000 \
    --voxel_size 0.0005 \
    --prune_ratio 1.0 \
    --start_stat 200 \
    --update_from 800 \
    --update_until 6000 \
    --update_interval 50 \
    --success_threshold 0.6 \
    --densify_grad_threshold 0.0001 \
    --min_opacity 0.002 \
    --mlp_opacity_lr_max_steps 8000 \
    --mlp_cov_lr_max_steps 8000 \
    --mlp_color_lr_max_steps 8000 \
    --mlp_color_lr_init 0.008 \
    --mlp_color_lr_final 0.00025 \
    --offset_lr_init 0.001 \
    --offset_lr_final 0.00001 \
    --feat_dim 32 \
    --reflectance_consistency_reg 2e-5 \
    --reflectance_smooth_reg 0.0 \
    --reflectance_edge_reg 2e-4 \
    --reflectance_edge_uplift_reg 3e-3 \
    --reflectance_contrast_reg 2e-3 \
    --reflectance_highfreq_reg 3e-3 \
    --highlight_reflectance_reg 1e-3 \
    --residual_chroma_reg 5e-4 \
    --reflectance_detail_reg 1e-6 \
    --reflectance_decoder_reg 2e-5 \
    --reflectance_offset_lr 0.008 \
    --reflectance_decoder_lr 0.002 \
    --b0_spatial_smooth_reg 0.0 \
    --residual_start_iter 3000 \
    --residual_ramp_iters 2500 \
    --enhancement_reflectance_reg 0.06 \
    --enhancement_degree_reg 0.2 \
    --enhancement_degree_global_reg 0.05 \
    --enhancement_smooth_reg 4e-4 \
    --enhancement_diff_start_iter 2350 \
    --enhancement_color_reg 0.06 \
    --enhancement_color_std_reg 0.02 \
    --enhancement_green_bias_reg 0.06 \
    --enhancement_prior cidnet \
    --cidnet_conda_env CIDNet \
    --cidnet_root ./submodules/HVI-CIDNet \
    --cidnet_weights ./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth \
    --cidnet_refresh_interval 0 \
    --cidnet_mlp_steps 100 \
    --cidnet_target_exposure 0.5 \
    --cidnet_refresh_reg 0.5 \
    --cidnet_color_reg 0.2 \
    --cidnet_param_reg 0.1 \
    --cidnet_mv_reg 0.5 \
    --cidnet_alpha_init 0.9 \
    --cidnet_gamma_init 0.9
