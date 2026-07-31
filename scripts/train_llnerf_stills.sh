#!/usr/bin/env bash
# Train LL-Gaussian on the LLNeRF still2--still4 low-light scenes.
# Normal-light images are linked only as evaluation GT and are never read by train.py.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/liuyuhao/datasets/llnerf-dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-/home/liuyuhao/datasets/normal-light-scenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuyuhao/outputs/llgaussian_llnerf}"
GPU="${GPU:-1}"
# train.py runs a fixed 2k warmup before this 8k main stage when --warmup is set.
ITERATIONS="${ITERATIONS:-8000}"
SCENES=(still2 still3 still4)

image_stems() {
    find "$1" -maxdepth 1 -type f -printf '%f\n' \
        | sed 's/\.[^.]*$//' \
        | LC_ALL=C sort -u
}

test_stems() {
    # Matches readColmapSceneInfo(..., eval=True, lod=0, llffhold=8): idx % 8 == 0.
    image_stems "$1" | awk 'NR % 8 == 1'
}

has_direct_images() {
    [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f -print -quit | grep -q .
}

resolve_normal_images() {
    local scene="$1"
    local candidate
    local candidates=(
        "$NORMAL_ROOT/$scene/images"
        "$NORMAL_ROOT/$scene"
        "$NORMAL_ROOT/images/$scene"
    )

    for candidate in "${candidates[@]}"; do
        if has_direct_images "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

prepare_gt_link() {
    local scene="$1"
    local low_scene="$DATA_ROOT/$scene"
    local normal_images
    local gt_link="$low_scene/gt"
    local missing_test_stems

    [[ -d "$low_scene/images" ]] || { echo "Missing low-light images: $low_scene/images" >&2; exit 1; }
    [[ -d "$low_scene/sparse" ]] || { echo "Missing COLMAP sparse data: $low_scene/sparse" >&2; exit 1; }
    normal_images="$(resolve_normal_images "$scene" || true)"
    if [[ -z "$normal_images" ]]; then
        echo "Could not locate normal-light images for $scene below $NORMAL_ROOT." >&2
        echo "Expected one of: $NORMAL_ROOT/$scene/images, $NORMAL_ROOT/$scene, or $NORMAL_ROOT/images/$scene" >&2
        echo "Candidate directories:" >&2
        find "$NORMAL_ROOT" -maxdepth 3 -type d \( -iname "$scene" -o -iname images \) -print >&2
        exit 1
    fi

    missing_test_stems="$(LC_ALL=C comm -23 <(test_stems "$low_scene/images") <(image_stems "$normal_images"))"
    [[ -z "$missing_test_stems" ]] || {
        echo "Normal-light GT is missing image stems for the test split of $scene:" >&2
        echo "$missing_test_stems" >&2
        exit 1
    }

    if [[ -e "$gt_link" || -L "$gt_link" ]]; then
        if [[ -L "$gt_link" && "$(readlink -f "$gt_link")" == "$(readlink -f "$normal_images")" ]]; then
            echo "GT link already correct: $gt_link"
        else
            echo "Refusing to overwrite existing GT path: $gt_link" >&2
            exit 1
        fi
    else
        ln -s "$normal_images" "$gt_link"
        echo "Linked GT: $gt_link -> $normal_images"
    fi
}

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

for scene in "${SCENES[@]}"; do
    prepare_gt_link "$scene"

    CUDA_VISIBLE_DEVICES="$GPU" python train.py \
        --eval \
        -s "$DATA_ROOT/$scene" \
        -m "$OUTPUT_ROOT/$scene" \
        --gpu "$GPU" \
        --iterations "$ITERATIONS" \
        --warmup \
        --use_wandb \
        --use_asg_illumination \
        --illumination_mode asg \
        --use_3D_filter \
        --use_residual \
        --use_dual_transient \
        --appearance_residual_dim 32
done
