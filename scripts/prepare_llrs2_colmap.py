"""Build an independent LLRS copy with LOM JSON and fresh COLMAP geometry."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def image_files(directory):
    return {path.name: path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def link_to(path, target):
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError(f"Unexpected existing symlink: {path}")
    elif path.exists():
        raise FileExistsError(path)
    else:
        path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def run(command, log, environment):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(map(str, command)) + "\n")
        handle.flush()
        subprocess.run(list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT,
                       env=environment, check=True)


def load_reader(project):
    spec = importlib.util.spec_from_file_location("llrs2_colmap_reader", project / "scene/colmap_loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_summary(path, reader):
    images = reader.read_extrinsics_binary(str(path / "images.bin"))
    with (path / "points3D.bin").open("rb") as handle:
        points = struct.unpack("<Q", handle.read(8))[0]
    return images, points


def stage_scene(source, output):
    low = image_files(source / "images")
    high = image_files(source / "gt/images")
    if not low or set(low) != set(high):
        raise ValueError(f"Unpaired source: {source}; low-only={set(low)-set(high)}, high-only={set(high)-set(low)}")
    if len({Path(name).stem for name in low}) != len(low):
        raise ValueError(f"Duplicate image stems: {source}")
    output.mkdir(exist_ok=True)
    records, dimensions = [], set()
    for name in sorted(low):
        record = {"name": name}
        for kind, files in (("low", low), ("high", high)):
            destination = output / kind / name
            destination.parent.mkdir(exist_ok=True)
            checksum = digest(files[name])
            if destination.exists():
                if digest(destination) != checksum:
                    raise ValueError(f"Refusing to replace different image: {destination}")
            else:
                shutil.copy2(files[name], destination)
                if digest(destination) != checksum:
                    raise ValueError(f"Copy checksum failed: {destination}")
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                size = image.size
            if kind == "low":
                dimensions.add(size)
                record["width"], record["height"] = size
            elif size != (record["width"], record["height"]):
                raise ValueError(f"Image pair dimensions differ: {name}")
            record[kind + "_sha256"] = checksum
        records.append(record)
    if len(dimensions) != 1:
        raise ValueError(f"Shared-camera reconstruction needs uniform dimensions: {source}")
    link_to(output / "images", output / "low")
    (output / "gt").mkdir(exist_ok=True)
    link_to(output / "gt/images", output / "high")
    return records


def export_scene(output, model, reader, args, environment, records):
    sparse = output / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin", "project.ini"):
        if (model / name).is_file():
            shutil.copy2(model / name, sparse / name)
    run([args.colmap, "model_converter", "--input_path", sparse, "--output_path", sparse / "points3D.ply",
         "--output_type", "PLY"], output / "colmap_logs/export.log", environment)
    text = output / "colmap_text"
    text.mkdir(exist_ok=True)
    run([args.colmap, "model_converter", "--input_path", sparse, "--output_path", text,
         "--output_type", "TXT"], output / "colmap_logs/export.log", environment)
    (output / "colmap_sparse").mkdir(exist_ok=True)
    link_to(output / "colmap_sparse/0", sparse)
    link_to(output / "points3d.ply", sparse / "points3D.ply")
    cameras = reader.read_intrinsics_binary(str(sparse / "cameras.bin"))
    images, point_count = model_summary(sparse, reader)
    expected = {item["name"] for item in records}
    registered = {image.name for image in images.values()}
    if registered != expected or point_count == 0 or len(cameras) != 1:
        raise ValueError(f"Incomplete geometry in {output}: missing={expected-registered}, points={point_count}")
    camera = next(iter(cameras.values()))
    if camera.model != "PINHOLE":
        raise ValueError(f"Unexpected final camera: {camera.model}")
    fx, fy, cx, cy = map(float, camera.params)
    if not np.isfinite(camera.params).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid calibrated intrinsics")
    header = dict(camera_angle_x=2 * math.atan(camera.width / (2 * fx)),
                  camera_angle_y=2 * math.atan(camera.height / (2 * fy)),
                  fl_x=fx, fl_y=fy, k1=0.0, k2=0.0, p1=0.0, p2=0.0,
                  cx=cx, cy=cy, w=float(camera.width), h=float(camera.height))
    train, test, full, pose_errors = [], [], [], []
    # Match the current project's default COLMAP split so JSON and COLMAP
    # consumers select the same held-out views. LOM JSON tests reference high.
    ordered = sorted(images.values(), key=lambda image: Path(image.name).stem)
    for index, image in enumerate(ordered):
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = image.qvec2rotmat()
        world_to_camera[:3, 3] = image.tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        camera_to_world[:3, 1:3] *= -1  # COLMAP camera axes -> OpenGL/NeRF axes.
        recovered = camera_to_world.copy()
        recovered[:3, 1:3] *= -1
        error = float(np.max(np.abs(np.linalg.inv(recovered) - world_to_camera)))
        pose_errors.append(error)
        if not np.isfinite(camera_to_world).all() or error > 1e-8:
            raise ValueError(f"Pose conversion failed: {image.name}")
        with Image.open(output / "high" / image.name) as photograph:
            gray = np.asarray(photograph.convert("L"), dtype=np.float32)
        laplacian = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
                     + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
        common = dict(sharpness=float(laplacian.var()), transform_matrix=camera_to_world.tolist())
        full.append(dict(file_path="low/" + image.name, **common))
        target = test if index % args.holdout == 0 else train
        target.append(dict(file_path=("high/" if target is test else "low/") + image.name, **common))
    for name, frames in (("train", train), ("test", test), ("val", test[:1]), ("all", full)):
        write_json(output / f"transforms_{name}.json", dict(header, frames=frames))
    report = dict(scene=output.name, status="complete", source=str(args.source / output.name),
                  geometry_input="all paired normal-exposure gt/images; fresh COLMAP SfM",
                  camera_model=camera.model, cameras=len(cameras), input_views=len(records),
                  registered_views=len(images), missing_views=sorted(expected-registered),
                  points3D=point_count, train_views=len(train), test_views=len(test), val_views=min(1, len(test)),
                  holdout_stride=args.holdout, split_order="lexicographic image stem; index modulo holdout == 0 is test",
                  coordinate_system="COLMAP world preserved; only NeRF camera Y/Z axes flipped",
                  max_pose_roundtrip_error=max(pose_errors), images=records,
                  colmap_executable=str(args.colmap), geometry_uses_normal_exposure=True,
                  original_geometry_or_priors_copied=False)
    write_json(output / "dataset_manifest.json", report)
    return report


def build_scene(source, output, reader, args, environment):
    records = stage_scene(source, output)
    logs = output / "colmap_logs"
    logs.mkdir(exist_ok=True)
    database = output / "colmap.db"
    # Source LLRS images are already undistorted. Fit fresh PINHOLE intrinsics
    # and poses; do not import the old reconstruction or neural point clouds.
    run([args.colmap, "feature_extractor", "--database_path", database,
         "--image_path", output / "high", "--ImageReader.camera_model", "PINHOLE",
         "--ImageReader.single_camera", "1", "--FeatureExtraction.use_gpu", "1",
         "--FeatureExtraction.gpu_index", "0", "--FeatureExtraction.num_threads", str(args.threads),
         "--SiftExtraction.max_num_features", "8192"], logs / "features.log", environment)
    run([args.colmap, "exhaustive_matcher", "--database_path", database,
         "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
         "--FeatureMatching.num_threads", str(args.threads), "--FeatureMatching.guided_matching", "1"],
        logs / "matching.log", environment)
    candidates = []
    for attempt, extra in (("standard", []), ("wide_baseline", ["--Mapper.init_min_tri_angle", "4"])):
        models = output / "_colmap_work" / attempt
        models.mkdir(parents=True, exist_ok=True)
        existing = list(models.glob("*/images.bin"))
        if not existing:
            run([args.colmap, "mapper", "--database_path", database, "--image_path", output / "high",
                 "--output_path", models, "--Mapper.num_threads", str(args.threads),
                 "--Mapper.random_seed", "0", "--Mapper.ba_use_gpu", "0", *extra],
                logs / (attempt + ".log"), environment)
        for image_file in models.glob("*/images.bin"):
            images, points = model_summary(image_file.parent, reader)
            candidates.append((len(images), points, image_file.parent))
        if candidates and max(item[0] for item in candidates) == len(records):
            break
    if not candidates:
        raise ValueError(f"COLMAP did not reconstruct any model: {source.name}")
    registered, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
    if registered != len(records):
        images, _ = model_summary(selected, reader)
        missing = sorted({item["name"] for item in records} - {image.name for image in images.values()})
        raise ValueError(f"Only {registered}/{len(records)} views registered: {missing}; no fabricated poses exported")
    return export_scene(output, selected, reader, args, environment, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=8)
    args = parser.parse_args()
    if args.holdout < 2 or args.threads < 1:
        parser.error("holdout must be >=2 and threads >=1")
    args.source = args.source.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output == args.source or args.source in args.output.parents:
        parser.error("Output must be independent of the source dataset")
    args.output.mkdir(exist_ok=True)
    marker = args.output / "build_config.json"
    config = dict(source=str(args.source), geometry_input="high", holdout=args.holdout,
                  format="LOM low/high+transforms JSON and LL-Gaussian COLMAP layout", gpu=args.gpu)
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise ValueError("Existing LLRS2 build configuration differs")
    write_json(marker, config)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, QT_QPA_PLATFORM="offscreen",
                       OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
    reader = load_reader(args.project)
    results = []
    started = time.time()
    for source in sorted(args.source.iterdir()):
        if not (source / "images").is_dir():
            continue
        output = args.output / source.name
        print("SCENE_START", source.name, flush=True)
        try:
            report = build_scene(source, output, reader, args, environment)
            print("SCENE_COMPLETE", source.name, report["registered_views"], report["points3D"], flush=True)
        except Exception as error:
            report = dict(scene=source.name, status="failed", error=str(error))
            print("SCENE_FAILED", source.name, repr(error), flush=True)
        results.append(report)
        write_json(args.output / "build_report.json", dict(elapsed_seconds=time.time()-started, scenes=results))
    complete = all(report["status"] == "complete" for report in results) and bool(results)
    print("BUILD_COMPLETE" if complete else "BUILD_INCOMPLETE", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
