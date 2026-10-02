#!/usr/bin/env bash
set -euo pipefail

SCENE_FILTER="chair"

usage() {
    echo "Usage: bash scripts/render_ablation_sharpen.sh [--scene all|buu|chair|sofa]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE_FILTER="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$SCENE_FILTER" in all|buu|chair|sofa) ;; *) echo "Invalid scene: $SCENE_FILTER" >&2; exit 2 ;; esac

VARIANTS=(a0_baseline no_contrast minimal_structure no_highfreq no_structure_all no_highlight no_tiny_regs isolate_off asg_lobes_2)

render_one() {
    local scene="$1"
    local variant="$2"
    local output="outputs/${scene}_sharpen_${variant}_8k"
    if [[ ! -d "$output" ]]; then
        echo "Missing experiment directory: $output" >&2
        return 1
    fi
    echo "Rendering scene=$scene variant=$variant model=$output"
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

for scene in "${SCENES[@]}"; do
    [[ "$SCENE_FILTER" != "all" && "$SCENE_FILTER" != "$scene" ]] && continue
    for variant in "${VARIANTS[@]}"; do
        render_one "$scene" "$variant"
    done
done

echo "All selected sharpen-ablation renders completed."

python scripts/collect_sharpen_ablation_results.py --scene "$SCENE_FILTER"
