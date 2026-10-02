"""Prepare a pinhole, scale-aware copy of one LLNeRF COLMAP scene.

The training loader ignores SIMPLE_RADIAL distortion.  This script remaps the
images to the original pinhole intrinsics and changes cameras.bin.  The copied
images.bin still has the original 2D feature tracks; train.py only reads its
camera poses and image names.  The source scene is read-only.
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_points3D_binary)


def undistort_image(source, target, camera):
    image = np.asarray(Image.open(source).convert("RGB"))
    if image.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"{source}: image size disagrees with cameras.bin")
    f, cx, cy, k1 = map(float, camera.params)
    intrinsic = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    distortion = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    output = cv2.undistort(image, intrinsic, distortion, None, intrinsic)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"quality": 95, "subsampling": 0} if target.suffix.lower() in (".jpg", ".jpeg") else {}
    Image.fromarray(output).save(target, **save_options)


def write_pinhole_cameras(path, cameras):
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            f, cx, cy, _ = map(float, camera.params)
            stream.write(struct.pack("<iiQQddd", camera.id, 0, camera.width,
                                     camera.height, f, cx, cy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="Scene name, for example shrub or room")
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/liuyuhao/datasets/llnerf-dataset"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("datasets/llnerf_prepared"))
    parser.add_argument("--enhanced-dir", type=Path,
                        help="Optional 2D enhancement images with the original camera geometry")
    parser.add_argument("--enhanced-already-undistorted", action="store_true",
                        help="Copy selected 2D targets without another radial remap")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.enhanced_dir:
        args.enhanced_dir = args.enhanced_dir.resolve()

    source = args.source_root / args.scene
    target = args.output_root / args.scene
    sparse = source / "sparse" / "0"
    cameras = read_intrinsics_binary(str(sparse / "cameras.bin"))
    if len(cameras) != 1:
        raise ValueError("This preparer currently supports one shared COLMAP camera")
    if any(camera.model != "SIMPLE_RADIAL" for camera in cameras.values()):
        raise ValueError("This preparer expects SIMPLE_RADIAL COLMAP cameras")
    if any(abs(float(camera.params[3])) > 0.5 for camera in cameras.values()):
        raise ValueError("Implausible radial coefficient; inspect the COLMAP reconstruction")
    if (target / "preparation.json").exists():
        raise FileExistsError(f"{target} is already prepared; choose a fresh output root")
    target_sparse = target / "sparse" / "0"
    target_sparse.mkdir(parents=True, exist_ok=True)
    for name in ("images.bin", "points3D.bin"):
        shutil.copy2(sparse / name, target_sparse / name)
    write_pinhole_cameras(target_sparse / "cameras.bin", cameras)

    camera = next(iter(cameras.values()))
    image_files = sorted(path for path in (source / "images").iterdir()
                         if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for image_path in image_files:
        undistort_image(image_path, target / "images" / image_path.name, camera)

    if args.enhanced_dir:
        prior_dir = target / "cidnet_prior" / "round_000" / "images"
        registered = read_extrinsics_binary(str(sparse / "images.bin"))
        names = sorted(Path(item.name).stem for item in registered.values())
        # The training command uses --eval and the loader's LLFF holdout of 8.
        train_names = {name for index, name in enumerate(names) if index % 8 != 0}
        for image_path in image_files:
            if image_path.stem not in train_names:
                continue
            matches = list(args.enhanced_dir.glob(image_path.stem + ".*"))
            if len(matches) != 1:
                raise ValueError(f"Expected one enhanced image for {image_path.stem}")
            target_image = prior_dir / (image_path.stem + ".png")
            if args.enhanced_already_undistorted:
                target_image.parent.mkdir(parents=True, exist_ok=True)
                selected_image = Image.open(matches[0]).convert("RGB")
                if selected_image.size != (camera.width, camera.height):
                    raise ValueError(f"{matches[0]}: size disagrees with cameras.bin")
                selected_image.save(target_image)
            else:
                undistort_image(matches[0], target_image, camera)
        (prior_dir.parent / "params.json").write_text('{"round": 0, "images": {}}\n')

    points, _, _ = read_points3D_binary(str(sparse / "points3D.bin"))
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    median_spacing = float(np.median(nearest))
    # Densification grids are 16x, 4x and 1x voxel_size in this project.
    voxel_size = median_spacing / 16.0
    report = {
        "source": str(source.resolve()),
        "images": len(image_files),
        "points": len(points),
        "radial_coefficients": [float(c.params[3]) for c in cameras.values()],
        "median_nearest_point_spacing": median_spacing,
        "recommended_voxel_size": voxel_size,
        "enhanced_images": str(args.enhanced_dir.resolve()) if args.enhanced_dir else None,
        "enhanced_already_undistorted": args.enhanced_already_undistorted,
    }
    (target / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
