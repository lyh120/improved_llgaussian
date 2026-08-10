#!/usr/bin/env bash
set -euo pipefail

# Recovery profile for still2.  It deliberately keeps the learned geometry
# unconstrained by the experimental scale guards and does not apply the 3D
# filter.  The previous guarded run saturated the scale cap at 0.016 and
# collapsed the rendered coverage to sparse points.
if [[ "${ENABLE_SCALE_GUARDS:-0}" == "1" ]]; then
    echo "ERROR: ENABLE_SCALE_GUARDS=1 is disabled for still2 recovery; it collapses scene coverage."
    echo "Run: bash scripts/run_still2_asg_safe_densify.sh"
    exit 2
fi
if [[ "${USE_3D_FILTER:-0}" == "1" ]]; then
    echo "ERROR: USE_3D_FILTER=1 is disabled for this still2 recovery experiment."
    echo "Run: bash scripts/run_still2_asg_safe_densify.sh"
    exit 2
fi

model_path=/home/liuyuhao/ll_further/LL-Gaussian-sg/experiments/still2_asg5_dense_recovery_v6

python /home/liuyuhao/ll_further/LL-Gaussian-sg/train.py \
    -s /home/liuyuhao/datasets/llnerf-dataset/still2 \
    -m "${model_path}" \
    --port 45710 \
    --eval \
    --gpu 1 \
    --use_wandb \
    --warmup \
    --use_asg_illumination \
    --illumination_mode asg \
    --asg_lobes 1 \
    --asg_lambda_min 1.0 \
    --asg_energy_reg 1e-4 \
    --asg_sharpness_reg 5e-5 \
    --asg_anisotropy_reg 1e-5 \
    --iterations 6000 \
    --save_iterations 3000 4000 5000 6000 \
    --test_iterations 3000 4000 5000 6000 \
    --position_lr_max_steps 6000 \
    --offset_lr_max_steps 6000 \
    --mlp_opacity_lr_max_steps 6000 \
    --mlp_cov_lr_init 0.001 \
    --mlp_cov_lr_final 0.0001 \
    --mlp_cov_lr_max_steps 6000 \
    --mlp_color_lr_max_steps 6000 \
    --mlp_color_lr_init 0.008 \
    --mlp_color_lr_final 0.00025 \
    --mlp_enhance_lr_init 0.02 \
    --mlp_enhance_lr_final 0.00025 \
    --scaling_lr 0.002 \
    --offset_lr_init 0.001 \
    --offset_lr_final 0.00001 \
    --voxel_size 0.0005 \
    --prune_ratio 1.0 \
    --feat_dim 32 \
    --start_stat 500 \
    --update_from 1200 \
    --update_until 4800 \
    --update_interval 100 \
    --success_threshold 0.8 \
    --densify_grad_threshold 0.00020 \
    --min_opacity 0.006 \
    --max_anchors 18000 \
    --max_new_anchors_per_update 320 \
    --densify_level_caps 160,100,60 \
    --prune_from_iter 1600 \
    --max_pruned_anchors_per_update 320 \
    --anchor_prune_grace_iters 700 \
    --warmup_start_stat 200 \
    --warmup_update_from 800 \
    --warmup_update_until 2000 \
    --warmup_update_interval 100 \
    --warmup_max_new_anchors 192 \
    --warmup_level_caps 96,60,36 \
    --warmup_densify_grad_threshold 0.00018 \
    --warmup_success_threshold 0.8 \
    --warmup_prune_from_iter 1200 \
    --warmup_max_pruned_anchors_per_update 128 \
    --illumination_smooth_reg 1e-4 \
    --illumination_smooth_kernel_size 5 \
    --warmup_illumination_smooth_reg 5e-5 \
    --warmup_illumination_smooth_kernel_size 9 \
    --reflectance_consistency_reg 2e-5 \
    --reflectance_smooth_reg 0.0 \
    --reflectance_edge_reg 2e-4 \
    --reflectance_edge_uplift_reg 3e-3 \
    --reflectance_contrast_reg 2e-3 \
    --reflectance_highfreq_reg 3e-3 \
    --highlight_reflectance_reg 1e-3 \
    --reflectance_detail_reg 1e-6 \
    --reflectance_decoder_reg 2e-5 \
    --reflectance_offset_lr 0.008 \
    --reflectance_decoder_lr 0.002 \
    --b0_spatial_smooth_reg 0.0 \
    --enhancement_diff_start_iter 3500 \
    --enhancement_smooth_reg 0.0 \
    --enhancement_gain_smooth_reg 5e-5 \
    --enhancement_edge_preserve_reg 0.02 \
    --enhancement_reflectance_reg 0.0 \
    --enhancement_degree_reg 0.12 \
    --enhancement_degree_global_reg 0.02 \
    --enhancement_color_reg 0.03 \
    --enhancement_color_std_reg 0.01 \
    --enhancement_green_bias_reg 0.03 \
    --enhancement_grad_clip 1.0 \
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
    --cidnet_alpha_init 1.4 \
    --cidnet_gamma_init 1.4 \
    --wandb_monitor_camera 1 \
    --wandb_monitor_split test \
    --wandb_monitor_interval 300 \
    --scale_stats_interval 300
