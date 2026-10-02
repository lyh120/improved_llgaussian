"""Prepare denoised, color-balanced CIDNet targets for LLNeRF scenes.

The source and output should contain registered, already undistorted PNGs.
This keeps multi-view geometry unchanged and only changes 2D teacher images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--denoise-strength", type=float, default=8.0)
    parser.add_argument("--grayworld-strength", type=float, default=0.8)
    parser.add_argument("--unsharp-amount", type=float, default=0.35)
    args = parser.parse_args()
    paths = sorted(args.source.glob("*.png"))
    if not paths:
        parser.error("No PNGs found in source")
    if args.destination.exists() and any(args.destination.iterdir()):
        parser.error("Destination must be empty")
    if args.denoise_strength <= 0 or not 0 <= args.grayworld_strength <= 1 or args.unsharp_amount < 0:
        parser.error("Invalid enhancement parameters")
    args.destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read {path}")
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr, None, args.denoise_strength, args.denoise_strength, 7, 21
        ).astype(np.float32) / 255.0
        # Gray-world uses only medium luminance pixels, so black background
        # and clipped highlights do not dominate the global white balance.
        value = denoised.mean(axis=2)
        mask = (value > 0.12) & (value < 0.85)
        mean = denoised[mask].mean(axis=0) if np.any(mask) else denoised.mean(axis=(0, 1))
        neutral = float(mean.mean())
        gain = np.power(neutral / np.maximum(mean, 0.02), args.grayworld_strength)
        gain = np.clip(gain, 0.8, 1.25)
        balanced = np.clip(denoised * gain.reshape(1, 1, 3), 0, 1)
        blur = cv2.GaussianBlur(balanced, (0, 0), 1.2)
        sharp = np.clip(balanced + args.unsharp_amount * (balanced - blur), 0, 1)
        output = np.round(sharp * 255).astype(np.uint8)
        if not cv2.imwrite(str(args.destination / path.name), output):
            raise OSError(f"Could not write {args.destination / path.name}")
    print(f"Saved {len(paths)} targets to {args.destination}")


if __name__ == "__main__":
    main()
