#!/usr/bin/env bash
set -euo pipefail

SCENE_FILTER="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE_FILTER="$2"; shift 2 ;;
        -h|--help) echo "Usage: bash scripts/render_rl_mlp_ablation.sh [--scene all|bike|buu|chair|sofa]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$SCENE_FILTER" in all|bike|buu|chair|sofa) ;; *) echo "Invalid scene: $SCENE_FILTER" >&2; exit 2 ;; esac

render_one() {
    local scene="$1"
    local output="$2"
    [[ -d "$output" ]] || { echo "Missing experiment directory: $output" >&2; return 1; }
    python render.py -m "$output" --dataset_path "datasets/$scene" --iteration 8000 \
        --skip_train --skip_optimize --profile_render_timing --profile_warmup_views 0
}

SCENES=(bike buu chair sofa)
VARIANTS=(no_residual mlp_r mlp_l mlp_rl)
for scene in "${SCENES[@]}"; do
    [[ "$SCENE_FILTER" != "all" && "$SCENE_FILTER" != "$scene" ]] && continue
    for variant in "${VARIANTS[@]}"; do
        render_one "$scene" "outputs/${scene}_ablation_${variant}_8k"
    done
done
python scripts/collect_rl_mlp_ablation_results.py --scene "$SCENE_FILTER"
