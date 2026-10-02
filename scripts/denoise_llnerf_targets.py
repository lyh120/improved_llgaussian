"""Denoise a completed CIDNet image set for --enhanced-dir experiments.

Example:
  python scripts/denoise_llnerf_targets.py \
    datasets/llnerf_prepared/_runs/room_detail12k_edge03/cidnet_prior/round_000/images \
    datasets/llnerf_targets/room_cidnet_nlm8 --strength 8
"""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--strength", type=float, default=8.0,
                        help="OpenCV colored nonlocal-means h in 8-bit pixel units")
    args = parser.parse_args()
    if args.strength <= 0:
        parser.error("--strength must be positive")
    paths = sorted(args.source.glob("*.png"))
    if not paths:
        parser.error(f"No PNG targets found in {args.source}")
    args.destination.mkdir(parents=True, exist_ok=True)
    existing = list(args.destination.iterdir())
    if existing:
        parser.error(f"Destination is not empty: {args.destination}")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, args.strength, args.strength, 7, 21
        )
        if not cv2.imwrite(str(args.destination / path.name), denoised):
            raise OSError(f"Could not write {args.destination / path.name}")
    print(f"Saved {len(paths)} targets to {args.destination}")


if __name__ == "__main__":
    main()
