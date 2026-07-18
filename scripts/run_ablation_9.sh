#!/usr/bin/env bash
set -euo pipefail

GPU="1"
SCENE_FILTER="all"
VARIANT_FILTER="all"
SKIP_EXISTING=0
MIN_FREE_GB=5

usage() {
    echo "Usage: bash scripts/run_ablation_9.sh [--gpu N] [--scene all|buu|chair|sofa] [--variant all|no_residual|sg|no_proposed_priors] [--skip-existing] [--min-free-gb N]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --scene) SCENE_FILTER="$2"; shift 2 ;;
        --variant) VARIANT_FILTER="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$SCENE_FILTER" in all|buu|chair|sofa) ;; *) echo "Invalid scene: $SCENE_FILTER" >&2; exit 2 ;; esac
case "$VARIANT_FILTER" in all|no_residual|sg|no_proposed_priors) ;; *) echo "Invalid variant: $VARIANT_FILTER" >&2; exit 2 ;; esac
[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || { echo "--min-free-gb must be a non-negative integer" >&2; exit 2; }

COMMON_ARGS=(
    --eval
    --gpu "$GPU"
    --use_3D_filter
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
    --cidnet_refresh_interval 0
    --cidnet_mlp_steps 100
    --cidnet_target_exposure 0.5
    --cidnet_refresh_reg 0.5
    --cidnet_color_reg 0.2
    --cidnet_param_reg 0.1
    --cidnet_mv_reg 0.5
)

ASG_ARGS=(
    --use_asg_illumination
    --illumination_mode asg
    --asg_lobes 1
    --asg_lambda_min 1.0
    --asg_energy_reg 1e-4
    --asg_sharpness_reg 5e-5
    --asg_anisotropy_reg 1e-5
)

SG_ARGS=(
    --use_sg_illumination
    --illumination_mode sg
    --sg_lobes 4
    --sg_lambda_min 1.0
    --sg_energy_reg 1e-4
    --sg_smooth_reg 5e-5
)

NO_PRIOR_ARGS=(
    --asg_energy_reg 0
    --asg_sharpness_reg 0
    --asg_anisotropy_reg 0
    --reflectance_consistency_reg 0
    --reflectance_smooth_reg 0
    --reflectance_edge_reg 0
    --reflectance_edge_uplift_reg 0
    --reflectance_contrast_reg 0
    --reflectance_highfreq_reg 0
    --highlight_reflectance_reg 0
    --reflectance_detail_reg 0
    --reflectance_decoder_reg 0
    --b0_spatial_smooth_reg 0
    --enhancement_reflectance_reg 0
    --enhancement_degree_reg 0
    --enhancement_degree_global_reg 0
    --enhancement_smooth_reg 0
    --enhancement_diff_start_iter 9000
    --enhancement_color_reg 0
    --enhancement_color_std_reg 0
    --enhancement_green_bias_reg 0
)

run_experiment() {
    local scene="$1"
    local variant="$2"
    local output="outputs/${scene}_ablation_${variant}_8k"
    local alpha_gamma="1.6"
    local port
    [[ "$scene" == "buu" ]] && alpha_gamma="1.2"
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')

    local available_kb
    local required_kb=$((MIN_FREE_GB * 1024 * 1024))
    available_kb=$(df -Pk . | awk 'NR == 2 {print $4}')
    if [[ ! "$available_kb" =~ ^[0-9]+$ || "$available_kb" -lt "$required_kb" ]]; then
        echo "Insufficient disk space: require at least ${MIN_FREE_GB} GiB free before starting an experiment." >&2
        df -h . >&2
        return 1
    fi

    if [[ -e "$output" ]]; then
        if [[ "$SKIP_EXISTING" -eq 1 ]]; then
            if [[ -f "$output/point_cloud/iteration_8000/point_cloud.ply" ]]; then
                echo "Skipping completed experiment: $output"
            else
                echo "Skipping existing experiment (completion marker not found): $output"
                echo "  Check this directory manually; it may contain an incomplete run."
            fi
            return 0
        fi
        echo "Refusing to overwrite existing experiment: $output" >&2
        echo "Use --skip-existing to leave it untouched and continue with the remaining experiments." >&2
        return 1
    fi

    local variant_args=()
    case "$variant" in
        no_residual)
            variant_args=("${ASG_ARGS[@]}")
            ;;
        sg)
            variant_args=("${SG_ARGS[@]}" --use_residual)
            ;;
        no_proposed_priors)
            variant_args=("${ASG_ARGS[@]}" "${NO_PRIOR_ARGS[@]}")
            ;;
    esac

    echo "============================================================"
    echo "Training scene=$scene variant=$variant output=$output gpu=$GPU port=$port"
    echo "CIDNet policy: fixed round_000 (--cidnet_refresh_interval 0, no force refresh)"
    echo "============================================================"
    python train.py \
        -s "datasets/$scene" \
        -m "$output" \
        --port "$port" \
        "${COMMON_ARGS[@]}" \
        "${variant_args[@]}" \
        --cidnet_alpha_init "$alpha_gamma" \
        --cidnet_gamma_init "$alpha_gamma"
}

SCENES=(buu chair sofa)
VARIANTS=(no_residual sg no_proposed_priors)

for scene in "${SCENES[@]}"; do
    [[ "$SCENE_FILTER" != "all" && "$SCENE_FILTER" != "$scene" ]] && continue
    for variant in "${VARIANTS[@]}"; do
        [[ "$VARIANT_FILTER" != "all" && "$VARIANT_FILTER" != "$variant" ]] && continue
        run_experiment "$scene" "$variant"
    done
done

echo "All selected ablation training runs completed."
