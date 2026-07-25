#!/usr/bin/env bash
# Sequential 8k training for buu, sofa, chair, bike, and shrub.
# New runs are residual-free; WandB uses each experiment directory as its run name.
set -euo pipefail

GPU="0"
SCENE_FILTER="all"
DATA_ROOT="datasets"
EXPERIMENT_ROOT="experiments"
RUN_TAG="asg_nor_8k_v1"
SKIP_EXISTING=false

usage() {
    cat <<'EOF'
Usage: bash scripts/run_five_scene_asg_nor_8k.sh [options]

Options:
  --gpu N                 GPU index passed to train.py (default: 0)
  --scene NAME            all|buu|sofa|chair|bike|shrub (default: all)
  --data-root PATH        Dataset root containing scene folders (default: datasets)
  --experiment-root PATH  Root directory for named experiment folders (default: experiments)
  --run-tag NAME          Suffix used in each experiment/WandB name (default: asg_nor_8k_v1)
  --skip-existing         Skip a scene when its iteration_8000 PLY already exists
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --scene) SCENE_FILTER="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

case "$SCENE_FILTER" in all|buu|sofa|chair|bike|shrub) ;; *) echo "Invalid scene: $SCENE_FILTER" >&2; exit 2 ;; esac

COMMON_ARGS=(
    --eval --gpu "$GPU" --use_wandb --warmup --use_3D_filter
    --use_asg_illumination --illumination_mode asg
    --asg_lobes 1 --asg_lambda_min 1.0 --asg_energy_reg 1e-4
    --asg_sharpness_reg 5e-5 --asg_anisotropy_reg 1e-5
    --iterations 8000 --save_iterations 8000 --test_iterations 4000 6000 8000
    --position_lr_max_steps 8000 --offset_lr_max_steps 8000
    --mlp_opacity_lr_max_steps 8000 --mlp_cov_lr_max_steps 8000 --mlp_color_lr_max_steps 8000
    --voxel_size 0.0005 --prune_ratio 1.0 --feat_dim 32
    --start_stat 200 --update_from 2200 --update_until 5000 --update_interval 100
    --success_threshold 0.7 --densify_grad_threshold 0.00025 --min_opacity 0.005
    --max_anchors 60000 --max_new_anchors_per_update 512 --densify_level_caps 256,160,96
    --anchor_prune_grace_iters 500
    --mlp_color_lr_init 0.008 --mlp_color_lr_final 0.00025
    --offset_lr_init 0.001 --offset_lr_final 0.00001
    --reflectance_consistency_reg 2e-5 --reflectance_smooth_reg 0.0
    --reflectance_edge_reg 2e-4 --reflectance_edge_uplift_reg 3e-3
    --reflectance_contrast_reg 2e-3 --reflectance_highfreq_reg 3e-3
    --highlight_reflectance_reg 1e-3 --reflectance_detail_reg 1e-6
    --reflectance_decoder_reg 2e-5 --reflectance_offset_lr 0.008 --reflectance_decoder_lr 0.002
    --b0_spatial_smooth_reg 0.0
    --enhancement_reflectance_reg 0.06 --enhancement_degree_reg 0.2
    --enhancement_degree_global_reg 0.05 --enhancement_smooth_reg 4e-4
    --enhancement_diff_start_iter 2350 --enhancement_color_reg 0.06
    --enhancement_color_std_reg 0.02 --enhancement_green_bias_reg 0.06
    --enhancement_prior cidnet --cidnet_conda_env CIDNet
    --cidnet_root ./submodules/HVI-CIDNet
    --cidnet_weights ./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth
    --cidnet_refresh_interval 0 --cidnet_mlp_steps 100 --cidnet_target_exposure 0.5
    --cidnet_refresh_reg 0.5 --cidnet_color_reg 0.2 --cidnet_param_reg 0.1 --cidnet_mv_reg 0.5
)

free_port() {
    python -c 'import socket; sock=socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()'
}

run_scene() {
    local scene="$1"
    local alpha_gamma="1.6"
    local experiment_name="${scene}_${RUN_TAG}"
    local model_path="${EXPERIMENT_ROOT}/${experiment_name}"
    local port
    local -a scene_args=("${COMMON_ARGS[@]}")

    [[ "$scene" == "bike" ]] && alpha_gamma="0.9"
    [[ "$scene" == "buu" ]] && alpha_gamma="1.2"

    if [[ "$scene" == "bike" ]]; then
        # Recover the old bike schedule's early, high-frequency detail search,
        # but retain deterministic Top-K selection and hard caps.  At most
        # 384 anchors are added per update and the whole model cannot exceed
        # 60k anchors, so this cannot reproduce the former point explosion.
        scene_args+=(
            --update_from 800 --update_until 6000 --update_interval 50
            --success_threshold 0.6 --densify_grad_threshold 0.0001 --min_opacity 0.002
            --max_anchors 60000 --max_new_anchors_per_update 384 --densify_level_caps 192,120,72
            --anchor_prune_grace_iters 500
            --illumination_smooth_reg 1e-4 --warmup_illumination_smooth_reg 2e-5 --illumination_smooth_kernel_size 5
            --enhancement_smooth_reg 1e-4 --enhancement_reflectance_reg 0.02
            --enhancement_degree_reg 0.08 --enhancement_degree_global_reg 0.02
            --enhancement_color_reg 0.03 --enhancement_color_std_reg 0.01 --enhancement_green_bias_reg 0.03
        )
    fi

    if [[ ! -d "${DATA_ROOT}/${scene}" ]]; then
        echo "Dataset not found: ${DATA_ROOT}/${scene}" >&2
        return 1
    fi
    if [[ -e "$model_path" ]]; then
        if [[ "$SKIP_EXISTING" == true && -f "${model_path}/point_cloud/iteration_8000/point_cloud.ply" ]]; then
            echo "Skipping completed experiment: ${experiment_name}"
            return 0
        fi
        echo "Refusing to overwrite existing experiment: ${model_path}" >&2
        return 1
    fi

    port=$(free_port)
    echo "Training ${scene}: experiment=${experiment_name}, gpu=${GPU}, port=${port}"
    python train.py -s "${DATA_ROOT}/${scene}" -m "$model_path" --port "$port" \
        "${scene_args[@]}" \
        --cidnet_alpha_init "$alpha_gamma" --cidnet_gamma_init "$alpha_gamma"
}

SCENES=(buu sofa chair bike shrub)
for scene in "${SCENES[@]}"; do
    [[ "$SCENE_FILTER" != "all" && "$SCENE_FILTER" != "$scene" ]] && continue
    run_scene "$scene"
done
