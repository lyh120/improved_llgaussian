#!/usr/bin/env bash
set -euo pipefail

SCENE_FILTER="all"
INCLUDE_FULL=1

usage() {
    echo "Usage: bash scripts/render_ablation_9.sh [--scene all|buu|chair|sofa] [--no-full]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE_FILTER="$2"; shift 2 ;;
        --no-full) INCLUDE_FULL=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$SCENE_FILTER" in all|buu|chair|sofa) ;; *) echo "Invalid scene: $SCENE_FILTER" >&2; exit 2 ;; esac

render_one() {
    local scene="$1"
    local output="$2"
    if [[ ! -d "$output" ]]; then
        echo "Missing experiment directory: $output" >&2
        return 1
    fi
    echo "Rendering scene=$scene model=$output"
    python render.py \
        -m "$output" \
        --dataset_path "datasets/$scene" \
        --iteration 8000 \
        --skip_train \
        --skip_optimize \
        --profile_render_timing \
        --profile_warmup_views 0
}

SCENES=(buu chair sofa)
VARIANTS=(no_residual sg no_proposed_priors)

for scene in "${SCENES[@]}"; do
    [[ "$SCENE_FILTER" != "all" && "$SCENE_FILTER" != "$scene" ]] && continue
    if [[ "$INCLUDE_FULL" -eq 1 ]]; then
        render_one "$scene" "outputs/${scene}_cidnet_asg_enh_8k_batch4_v2"
    fi
    for variant in "${VARIANTS[@]}"; do
        render_one "$scene" "outputs/${scene}_ablation_${variant}_8k"
    done
done

echo "All selected ablation renders completed."

python scripts/collect_ablation_results.py --scene "$SCENE_FILTER"
