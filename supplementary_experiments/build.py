#!/usr/bin/env python3
"""Build traceable supplementary figures and tables from existing experiment outputs.

The script is intentionally conservative: it reads result folders and writes only
inside ``supplementary_experiments/{data,figures,tables,report}``. Missing files
become explicit completeness-report entries; no metric is inferred from filenames.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is supplied by torchvision environments.
    Image = None


TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parent
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")
HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PLOT_ERROR: str | None = None
PLOTS_DISABLED = False
PLOTTER: Any | None = None


def get_plotter():
    """Load Matplotlib only when a figure is actually requested.

    This keeps the evidence tables usable in environments where a local Python
    installation has a NumPy/Matplotlib ABI mismatch. The project training
    environment declares Matplotlib in ``requirements.txt`` and will still
    produce all figures normally.
    """
    global PLOT_ERROR, PLOTTER
    if PLOTS_DISABLED:
        return None
    if PLOTTER is not None:
        return PLOTTER
    if PLOT_ERROR is not None:
        return None
    try:
        # Some broken binary builds print a full ABI traceback before raising an
        # ImportError. Keep the command output readable and record the failure
        # in completeness_report.json instead.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plotter

        PLOTTER = plotter
        return PLOTTER
    except Exception as error:  # pragma: no cover - depends on local Python ABI.
        PLOT_ERROR = f"{type(error).__name__}: {error}"
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "Manifest is not JSON-compatible YAML and PyYAML is unavailable. "
                "Either install PyYAML or keep the manifest in JSON-compatible YAML."
            ) from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError("Manifest must be a mapping containing an 'experiments' list.")
    return payload


def resolve_path(value: str | Path | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def safe_load_json(path: Path, completeness: list[dict[str, str]], experiment_id: str, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        completeness.append({"experiment_id": experiment_id, "artifact": label, "path": str(path), "status": "missing"})
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        completeness.append({"experiment_id": experiment_id, "artifact": label, "path": str(path), "status": f"unreadable: {error}"})
        return None


def find_iteration_dir(output_dir: Path, requested: Any, completeness: list[dict[str, str]], experiment_id: str) -> tuple[Path | None, int | None]:
    test_root = output_dir / "test"
    if not test_root.is_dir():
        completeness.append({"experiment_id": experiment_id, "artifact": "test results", "path": str(test_root), "status": "missing"})
        return None, None
    if requested not in (None, "latest", -1, "-1"):
        try:
            iteration = int(requested)
        except (TypeError, ValueError):
            completeness.append({"experiment_id": experiment_id, "artifact": "iteration", "path": str(test_root), "status": f"invalid iteration: {requested}"})
            return None, None
        result_dir = test_root / f"ours_{iteration}"
        if not result_dir.is_dir():
            completeness.append({"experiment_id": experiment_id, "artifact": "iteration result", "path": str(result_dir), "status": "missing"})
            return None, iteration
        return result_dir, iteration

    candidates: list[tuple[int, Path]] = []
    for candidate in test_root.glob("ours_*"):
        match = re.fullmatch(r"ours_(\d+)", candidate.name)
        if candidate.is_dir() and match:
            candidates.append((int(match.group(1)), candidate))
    if not candidates:
        completeness.append({"experiment_id": experiment_id, "artifact": "latest result", "path": str(test_root), "status": "no ours_<iteration> directory"})
        return None, None
    return max(candidates, key=lambda item: item[0])[1], max(candidates, key=lambda item: item[0])[0]


def count_ply_vertices(path: Path) -> int | None:
    """Read only a PLY header to obtain the number of stored anchors/vertices."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii", errors="ignore").strip()
                match = re.fullmatch(r"element\s+vertex\s+(\d+)", line)
                if match:
                    return int(match.group(1))
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def discover_anchor_count(output_dir: Path | None, iteration: int | None) -> tuple[int | None, Path | None]:
    if output_dir is None:
        return None, None
    candidates: list[Path] = []
    if iteration is not None:
        candidates.extend(
            [
                output_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                output_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply.gz",
            ]
        )
    candidates.extend(sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"), reverse=True))
    for candidate in candidates:
        count = count_ply_vertices(candidate)
        if count is not None:
            return count, candidate
    return None, None


def parse_training_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {"psnr": [], "anchors": [], "updates": []}
    if not path.is_file():
        return curves
    psnr_pattern = re.compile(r"\[ITER\s+(\d+)\]\s+Evaluating\s+(test|train).*?PSNR\s+([-+0-9.eE]+)")
    anchor_pattern = re.compile(r"(?:points number:|Number of points at initialisation\s*:)\s*(\d+)")
    update_pattern = re.compile(
        r"anchors_before\s*[=:]\s*(\d+).*?candidates\s*[=:]\s*(\d+).*?added_by_level\s*[=: ]\s*([^,\]\n]+).*?pruned\s*[=:]\s*(\d+).*?anchors_after\s*[=:]\s*(\d+)",
        re.IGNORECASE,
    )
    for line_index, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        match = psnr_pattern.search(raw_line)
        if match:
            curves["psnr"].append({"iteration": int(match.group(1)), "split": match.group(2), "value": float(match.group(3)), "line": line_index})
        match = anchor_pattern.search(raw_line)
        if match:
            curves["anchors"].append({"event_index": len(curves["anchors"]) + 1, "value": int(match.group(1)), "line": line_index})
        match = update_pattern.search(raw_line)
        if match:
            curves["updates"].append(
                {
                    "line": line_index,
                    "anchors_before": int(match.group(1)),
                    "candidates": int(match.group(2)),
                    "added_by_level": match.group(3).strip(),
                    "pruned": int(match.group(4)),
                    "anchors_after": int(match.group(5)),
                }
            )
    return curves


def image_files(directory: Path | None) -> dict[str, Path]:
    if directory is None or not directory.is_dir():
        return {}
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def collect_experiment(entry: dict[str, Any], outputs_root: Path, completeness: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    experiment_id = str(entry.get("id", "unnamed"))
    output_dir = resolve_path(entry.get("output_dir"), outputs_root)
    result_dir: Path | None = None
    iteration: int | None = None
    if entry.get("result_dir"):
        result_dir = resolve_path(entry["result_dir"], output_dir or outputs_root)
        match = re.search(r"ours_(\d+)", result_dir.name if result_dir else "")
        iteration = int(match.group(1)) if match else None
        if result_dir is not None and not result_dir.is_dir():
            completeness.append({"experiment_id": experiment_id, "artifact": "explicit result_dir", "path": str(result_dir), "status": "missing"})
            result_dir = None
    elif output_dir is not None:
        result_dir, iteration = find_iteration_dir(output_dir, entry.get("iteration", "latest"), completeness, experiment_id)

    record: dict[str, Any] = {
        "id": experiment_id,
        "scene": str(entry.get("scene", "unknown")),
        "display_name": str(entry.get("display_name", experiment_id)),
        "role": str(entry.get("role", "baseline")),
        "enabled": bool(entry.get("enabled", True)),
        "output_dir": str(output_dir) if output_dir else None,
        "result_dir": str(result_dir) if result_dir else None,
        "iteration": iteration,
        "notes": str(entry.get("notes", "")),
        "peak_memory_gb": entry.get("peak_memory_gb"),
        "metrics": {},
        "metric_sources": {},
        "timing": {},
        "timing_source": None,
        "anchor_count": None,
        "anchor_source": None,
        "diagnostics": {},
        "diagnostic_sources": {},
        "image_dirs": {},
        "log_path": None,
    }
    per_view_rows: list[dict[str, Any]] = []

    if result_dir is not None:
        for report_name in ("metrics_enhanced_gt", "metrics_lowlight"):
            metric_path = result_dir / f"{report_name}.json"
            payload = safe_load_json(metric_path, completeness, experiment_id, report_name)
            if not payload:
                continue
            summary = payload.get("summary", {})
            valid = {key: float(summary[key]) for key in METRIC_KEYS if key in summary and isinstance(summary[key], (int, float))}
            if valid:
                record["metrics"][report_name] = valid
                record["metric_sources"][report_name] = str(metric_path)
            for view_name, view_metrics in payload.get("per_view", {}).items():
                if not isinstance(view_metrics, dict):
                    continue
                row = {"experiment_id": experiment_id, "scene": record["scene"], "display_name": record["display_name"], "role": record["role"], "report": report_name, "view": str(view_name), "source": str(metric_path)}
                row.update({key: float(view_metrics[key]) for key in METRIC_KEYS if key in view_metrics and isinstance(view_metrics[key], (int, float))})
                per_view_rows.append(row)

        timing_path = result_dir / "render_timing_profile.json"
        timing_payload = safe_load_json(timing_path, completeness, experiment_id, "render timing")
        if timing_payload:
            record["timing"] = timing_payload.get("summary", {})
            record["timing_source"] = str(timing_path)
        for diagnostic_name in ("coverage_stats", "illumination_stats", "asg_stats", "sg_stats"):
            diagnostic_path = result_dir / f"{diagnostic_name}.json"
            payload = safe_load_json(diagnostic_path, completeness, experiment_id, diagnostic_name)
            if payload:
                record["diagnostics"][diagnostic_name] = payload.get("summary", payload)
                record["diagnostic_sources"][diagnostic_name] = str(diagnostic_path)
        for key, directory_name in {
            "renders": "renders",
            "enhanceds": "render_enhanceds",
            "ground_truth": "gt",
            "reflectances": "render_reflectances",
            "illuminations": "render_illuminations",
            "coverages": "render_coverages",
        }.items():
            path = result_dir / directory_name
            if path.is_dir():
                record["image_dirs"][key] = str(path)

    anchor_count, anchor_path = discover_anchor_count(output_dir, iteration)
    if anchor_count is not None:
        record["anchor_count"] = anchor_count
        record["anchor_source"] = str(anchor_path)
    elif output_dir is not None:
        completeness.append({"experiment_id": experiment_id, "artifact": "anchor PLY", "path": str(output_dir / "point_cloud"), "status": "missing"})

    log_path = resolve_path(entry.get("log_path"), REPO_ROOT)
    if log_path:
        record["log_path"] = str(log_path)
        if not log_path.is_file():
            completeness.append({"experiment_id": experiment_id, "artifact": "training log", "path": str(log_path), "status": "missing"})
    curves = parse_training_log(log_path) if log_path else {"psnr": [], "anchors": [], "updates": []}
    return record, per_view_rows, curves


def metric_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for report, values in record["metrics"].items():
            row = {"experiment_id": record["id"], "scene": record["scene"], "method": record["display_name"], "role": record["role"], "report": report, "iteration": record["iteration"], "source": record["metric_sources"].get(report, "")}
            row.update({key: values.get(key) for key in METRIC_KEYS})
            rows.append(row)
    return rows


def write_summary_plots(summary_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    paths: list[Path] = []
    plt = get_plotter()
    if plt is None:
        return paths
    enhanced_rows = [row for row in summary_rows if row["report"] == "metrics_enhanced_gt"]
    if not enhanced_rows:
        return paths
    scenes = sorted({row["scene"] for row in enhanced_rows})
    methods = list(dict.fromkeys(row["method"] for row in enhanced_rows))
    for metric in METRIC_KEYS:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(scenes)), 4.5), constrained_layout=True)
        x = np.arange(len(scenes))
        width = 0.8 / max(1, len(methods))
        drew = False
        for method_index, method in enumerate(methods):
            values = []
            for scene in scenes:
                matching = [row.get(metric) for row in enhanced_rows if row["scene"] == scene and row["method"] == method and row.get(metric) is not None]
                values.append(float(np.mean(matching)) if matching else np.nan)
            if np.all(np.isnan(values)):
                continue
            drew = True
            ax.bar(x - 0.4 + width / 2 + method_index * width, values, width, label=method)
        if drew:
            ax.set_xticks(x, scenes)
            ax.set_ylabel(metric)
            ax.set_title(f"Enhanced-GT {metric} by scene")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            path = figures_dir / f"enhanced_{metric.lower()}_by_scene.png"
            fig.savefig(path, dpi=220)
            paths.append(path)
        plt.close(fig)
    return paths


def bootstrap_mean_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    estimates = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_statistics(per_view_rows: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    by_scene_report: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_view_rows:
        by_scene_report[(row["scene"], row["report"])].append(row)
    rng = np.random.default_rng(20260725)
    rows: list[dict[str, Any]] = []
    for (scene, report), group in by_scene_report.items():
        main_rows = [row for row in group if row["role"] == "main"]
        if not main_rows:
            continue
        main_method = main_rows[0]["display_name"]
        by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            by_method[row["display_name"]][row["view"]] = row
        reference = by_method[main_method]
        for comparator, comparator_rows in by_method.items():
            if comparator == main_method:
                continue
            common_views = sorted(set(reference) & set(comparator_rows))
            for metric in METRIC_KEYS:
                deltas = np.array(
                    [float(reference[view][metric]) - float(comparator_rows[view][metric]) for view in common_views if metric in reference[view] and metric in comparator_rows[view]],
                    dtype=float,
                )
                if not len(deltas):
                    continue
                signed = deltas if HIGHER_IS_BETTER[metric] else -deltas
                ci_low, ci_high = bootstrap_mean_interval(signed, samples, rng)
                rows.append(
                    {
                        "scene": scene,
                        "report": report,
                        "reference": main_method,
                        "comparator": comparator,
                        "metric": metric,
                        "matched_views": len(signed),
                        "mean_delta_reference_minus_comparator": float(np.mean(signed)),
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "reference_win_rate": float(np.mean(signed > 0)),
                        "tie_rate": float(np.mean(np.isclose(signed, 0.0))),
                    }
                )
    return rows


def write_per_view_plots(per_view_rows: list[dict[str, Any]], statistics: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    paths: list[Path] = []
    plt = get_plotter()
    if plt is None:
        return paths
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_view_rows:
        if row["report"] == "metrics_enhanced_gt":
            for metric in METRIC_KEYS:
                if metric in row:
                    grouped[(row["scene"], metric, row["report"])].append(row)
    for (scene, metric, _), group in grouped.items():
        methods = list(dict.fromkeys(row["display_name"] for row in group))
        series = [[float(row[metric]) for row in group if row["display_name"] == method and metric in row] for method in methods]
        series = [values for values in series if values]
        labels = [method for method in methods if any(row["display_name"] == method and metric in row for row in group)]
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(labels)), 4.4), constrained_layout=True)
        violin = ax.violinplot(series, showmeans=True, showextrema=True)
        for body in violin["bodies"]:
            body.set_alpha(0.65)
        ax.boxplot(series, widths=0.16, showfliers=False, medianprops={"color": "black"})
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=16, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{scene}: per-view enhanced-GT {metric}")
        ax.grid(axis="y", alpha=0.25)
        path = figures_dir / f"per_view_{scene}_{metric.lower()}.png"
        fig.savefig(path, dpi=220)
        paths.append(path)
        plt.close(fig)

    if statistics:
        fig, ax = plt.subplots(figsize=(max(7, 0.95 * len(statistics)), 4.5), constrained_layout=True)
        labels = [f"{item['scene']}\n{item['metric']}\nvs {item['comparator']}" for item in statistics]
        means = [item["mean_delta_reference_minus_comparator"] for item in statistics]
        lows = [item["bootstrap_ci95_low"] for item in statistics]
        highs = [item["bootstrap_ci95_high"] for item in statistics]
        yerr = np.array([[mean - low if not math.isnan(low) else 0.0 for mean, low in zip(means, lows)], [high - mean if not math.isnan(high) else 0.0 for mean, high in zip(means, highs)]])
        ax.errorbar(np.arange(len(statistics)), means, yerr=yerr, fmt="o", capsize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(statistics)), labels, rotation=26, ha="right", fontsize=8)
        ax.set_ylabel("Signed delta (main method better > 0)")
        ax.set_title("Paired per-view effect with bootstrap 95% CI")
        ax.grid(axis="y", alpha=0.25)
        path = figures_dir / "paired_per_view_effects.png"
        fig.savefig(path, dpi=220)
        paths.append(path)
        plt.close(fig)
    return paths


def write_efficiency_table_and_plot(records: list[dict[str, Any]], tables_dir: Path, figures_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        enhanced = record["metrics"].get("metrics_enhanced_gt", {})
        timing = record.get("timing", {})
        if not enhanced and not timing and record.get("anchor_count") is None:
            continue
        rows.append(
            {
                "experiment_id": record["id"],
                "scene": record["scene"],
                "method": record["display_name"],
                "role": record["role"],
                "PSNR": enhanced.get("PSNR"),
                "SSIM": enhanced.get("SSIM"),
                "LPIPS": enhanced.get("LPIPS"),
                "render_core_fps": timing.get("render_core_time_fps"),
                "loop_wall_fps": timing.get("loop_wall_time_fps"),
                "anchor_count": record.get("anchor_count"),
                "peak_memory_gb": record.get("peak_memory_gb"),
                "metric_source": record["metric_sources"].get("metrics_enhanced_gt", ""),
                "timing_source": record.get("timing_source") or "",
                "anchor_source": record.get("anchor_source") or "",
            }
        )
    write_csv(tables_dir / "efficiency.csv", rows, list(rows[0]) if rows else ["experiment_id", "scene", "method", "role", "PSNR", "SSIM", "LPIPS", "render_core_fps", "loop_wall_fps", "anchor_count", "peak_memory_gb", "metric_source", "timing_source", "anchor_source"])
    paths: list[Path] = []
    plt = get_plotter()
    if plt is None:
        return rows, paths
    points = [row for row in rows if row.get("PSNR") is not None and row.get("render_core_fps") not in (None, 0)]
    if points:
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        colors = {"main": "tab:orange", "baseline": "tab:blue", "ablation": "tab:green", "diagnostic": "tab:gray"}
        for row in points:
            size = 45 if not row.get("anchor_count") else max(35, min(260, float(row["anchor_count"]) / 300))
            ax.scatter(row["render_core_fps"], row["PSNR"], s=size, color=colors.get(row["role"], "tab:gray"), alpha=0.78)
            ax.annotate(f"{row['scene']}: {row['method']}", (row["render_core_fps"], row["PSNR"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Core render FPS")
        ax.set_ylabel("Enhanced-GT PSNR")
        ax.set_title("Quality–speed evidence (marker size = anchor count when available)")
        ax.grid(alpha=0.25)
        path = figures_dir / "quality_speed_pareto.png"
        fig.savefig(path, dpi=220)
        paths.append(path)
        plt.close(fig)
    return rows, paths


def write_convergence_plots(curves_by_id: dict[str, dict[str, list[dict[str, Any]]]], records: list[dict[str, Any]], figures_dir: Path, data_dir: Path) -> list[Path]:
    paths: list[Path] = []
    record_by_id = {record["id"]: record for record in records}
    curve_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    plt = get_plotter()
    for experiment_id, curves in curves_by_id.items():
        record = record_by_id[experiment_id]
        for item in curves["psnr"]:
            curve_rows.append({"experiment_id": experiment_id, "scene": record["scene"], "method": record["display_name"], "curve": "PSNR", **item})
        for item in curves["anchors"]:
            curve_rows.append({"experiment_id": experiment_id, "scene": record["scene"], "method": record["display_name"], "curve": "anchors", **item})
        for item in curves["updates"]:
            update_rows.append({"experiment_id": experiment_id, "scene": record["scene"], "method": record["display_name"], **item})
        if plt is None or (not curves["psnr"] and not curves["anchors"]):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for split in ("test", "train"):
            points = [item for item in curves["psnr"] if item["split"] == split]
            if points:
                axes[0].plot([item["iteration"] for item in points], [item["value"] for item in points], marker="o", label=split)
        axes[0].set_title("Evaluation PSNR")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("PSNR")
        axes[0].grid(alpha=0.25)
        if axes[0].lines:
            axes[0].legend()
        if curves["anchors"]:
            axes[1].plot([item["event_index"] for item in curves["anchors"]], [item["value"] for item in curves["anchors"]], marker=".")
        axes[1].set_title("Anchor count from log events")
        axes[1].set_xlabel("Logged anchor event")
        axes[1].set_ylabel("Anchors")
        axes[1].grid(alpha=0.25)
        fig.suptitle(f"{record['scene']}: {record['display_name']}")
        path = figures_dir / f"convergence_{experiment_id}.png"
        fig.savefig(path, dpi=220)
        paths.append(path)
        plt.close(fig)
    write_csv(data_dir / "training_curves.csv", curve_rows, ["experiment_id", "scene", "method", "curve", "iteration", "split", "value", "event_index", "line"])
    write_csv(data_dir / "anchor_update_events.csv", update_rows, ["experiment_id", "scene", "method", "line", "anchors_before", "candidates", "added_by_level", "pruned", "anchors_after"])
    return paths


def load_image(path: Path) -> np.ndarray | None:
    if Image is None or not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def error_image(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        image = Image.fromarray(target).resize((prediction.shape[1], prediction.shape[0])) if Image else None
        if image is None:
            return np.zeros_like(prediction)
        target = np.asarray(image)
    return np.abs(prediction.astype(np.float32) - target.astype(np.float32)).mean(axis=2)


def write_qualitative_figures(records: list[dict[str, Any]], manifest: dict[str, Any], figures_dir: Path) -> list[Path]:
    plt = get_plotter()
    if Image is None or plt is None:
        return []
    paths: list[Path] = []
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["image_dirs"].get("enhanceds"):
            by_scene[record["scene"]].append(record)
    requested_views = manifest.get("qualitative_views", {})
    for scene, scene_records in by_scene.items():
        enhanced_maps = {record["id"]: image_files(Path(record["image_dirs"]["enhanceds"])) for record in scene_records}
        common = set.intersection(*(set(mapping) for mapping in enhanced_maps.values())) if enhanced_maps else set()
        view_names = requested_views.get(scene) or sorted(common)[:1]
        if not view_names:
            continue
        main_record = next((record for record in scene_records if record["role"] == "main" and record["image_dirs"].get("ground_truth")), scene_records[0])
        gt_map = image_files(Path(main_record["image_dirs"].get("ground_truth", "")))
        lowlight_map = image_files(Path(main_record["image_dirs"].get("renders", "")))
        for view_name in view_names:
            columns: list[tuple[str, np.ndarray | None]] = [("GT", load_image(gt_map[view_name]) if view_name in gt_map else None), ("Low-light render", load_image(lowlight_map[view_name]) if view_name in lowlight_map else None)]
            for record in scene_records:
                columns.append((record["display_name"], load_image(enhanced_maps[record["id"]][view_name]) if view_name in enhanced_maps[record["id"]] else None))
            if not any(image is not None for _, image in columns):
                continue
            gt = columns[0][1]
            fig, axes = plt.subplots(2, len(columns), figsize=(3.0 * len(columns), 5.5), constrained_layout=True)
            axes = np.atleast_2d(axes)
            for index, (label, image) in enumerate(columns):
                if image is not None:
                    axes[0, index].imshow(image)
                axes[0, index].set_title(label, fontsize=9)
                axes[0, index].axis("off")
                if image is not None and gt is not None and index >= 2:
                    axes[1, index].imshow(error_image(image, gt), cmap="magma", vmin=0, vmax=80)
                    axes[1, index].set_title("Absolute error", fontsize=9)
                axes[1, index].axis("off")
            fig.suptitle(f"{scene} · view {view_name}")
            path = figures_dir / f"qualitative_{scene}_{view_name}.png"
            fig.savefig(path, dpi=220)
            paths.append(path)
            plt.close(fig)

        for record in scene_records:
            if record["role"] != "main":
                continue
            directories = record["image_dirs"]
            diagnostic_keys = [("renders", "Low-light render"), ("enhanceds", "Enhanced"), ("reflectances", "Reflectance"), ("illuminations", "Illumination"), ("coverages", "Coverage")]
            diagnostic_maps = [(label, image_files(Path(directories[key]))) for key, label in diagnostic_keys if key in directories]
            if len(diagnostic_maps) < 3:
                continue
            name = (requested_views.get(scene) or sorted(next(iter(diagnostic_maps))[1])[:1])[0]
            fig, axes = plt.subplots(1, len(diagnostic_maps), figsize=(3 * len(diagnostic_maps), 3.2), constrained_layout=True)
            for axis, (label, mapping) in zip(np.atleast_1d(axes), diagnostic_maps):
                image = load_image(mapping[name]) if name in mapping else None
                if image is not None:
                    axis.imshow(image)
                axis.set_title(label, fontsize=9)
                axis.axis("off")
            fig.suptitle(f"{scene} diagnostics · {record['display_name']} · view {name}")
            path = figures_dir / f"diagnostics_{scene}_{record['id']}_{name}.png"
            fig.savefig(path, dpi=220)
            paths.append(path)
            plt.close(fig)
    return paths


def latex_value(value: Any, digits: int = 3) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.{digits}f}"


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("&", "\\&")


def write_latex_tables(summary_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, Any]], tables_dir: Path) -> None:
    def table(rows: list[list[str]], header: list[str], caption: str, label: str) -> str:
        alignment = "l" * len(header)
        body = "\n".join(" & ".join(row) + r" \\" for row in rows)
        return "\n".join([r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{alignment}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule", body or r"\multicolumn{1}{c}{No available evidence} \\", r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    enhanced = [row for row in summary_rows if row["report"] == "metrics_enhanced_gt"]
    metric_rows = [[latex_escape(row["scene"]), latex_escape(row["method"]), latex_value(row.get("PSNR"), 3), latex_value(row.get("SSIM"), 4), latex_value(row.get("LPIPS"), 4)] for row in enhanced]
    (tables_dir / "main_metrics.tex").write_text(table(metric_rows, ["Scene", "Method", "PSNR$\\uparrow$", "SSIM$\\uparrow$", "LPIPS$\\downarrow$"], "Enhanced-image metrics from registered result files.", "tab:supp_metrics"), encoding="utf-8")
    efficiency_table_rows = [[latex_escape(row["scene"]), latex_escape(row["method"]), latex_value(row.get("PSNR"), 3), latex_value(row.get("render_core_fps"), 2), "—" if row.get("anchor_count") is None else f"{int(row['anchor_count']):,}", latex_value(row.get("peak_memory_gb"), 2)] for row in efficiency_rows]
    (tables_dir / "efficiency.tex").write_text(table(efficiency_table_rows, ["Scene", "Method", "PSNR", "Core FPS", "Anchors", "Memory (GB)"], "Quality and efficiency evidence from registered results.", "tab:supp_efficiency"), encoding="utf-8")


def write_report(records: list[dict[str, Any]], completeness: list[dict[str, str]], figures: list[Path], summary_rows: list[dict[str, Any]], statistics: list[dict[str, Any]], report_dir: Path) -> None:
    available_metrics = len([row for row in summary_rows if row["report"] == "metrics_enhanced_gt"])
    lines = [
        "# Supplementary Experimental Evidence",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Protocol and provenance",
        "",
        "This supplement was generated only from the registered artifact paths. Missing files are reported explicitly; no missing metric or comparison has been inferred.",
        "",
        f"Registered experiments: {len(records)}. Enhanced-GT metric records: {available_metrics}. Missing/unreadable artifacts: {len(completeness)}.",
        "",
        "## Per-view statistics",
        "",
        "Paired win rates and bootstrap intervals are emitted only when a main-method record and a comparison share exact view identifiers. No paired claim is made for unmatched views.",
        "",
        f"Available paired comparisons: {len(statistics)}. See `data/per_view_statistics.csv`.",
        "",
        "## Efficiency and convergence",
        "",
        "Quality–speed plots use `render_core_time_fps` when a render timing profile exists. Anchor counts are read from PLY headers or omitted. Training curves are parsed from configured logs and distinguish iteration-indexed PSNR from event-indexed anchor logs.",
        "",
        "## Qualitative and diagnostic material",
        "",
        "Qualitative sheets are generated only from aligned image names. Diagnostic sheets require the corresponding reflectance, illumination, and coverage folders and are not substituted with synthetic visualizations.",
        "",
        "## Limitations",
        "",
        "This evidence package does not establish ASG-vs-SG, prior-splitting, seed robustness, or cross-view-warping claims unless matching registered outputs are supplied. Those comparisons remain outside the current evidence set.",
        "",
        "## Generated artifacts",
        "",
    ]
    lines.extend([f"- `{path.relative_to(TOOL_ROOT).as_posix()}`" for path in figures])
    (report_dir / "supplementary_experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = "\n".join(
        [
            r"\section{Supplementary Experimental Evidence}",
            "All quantities in this supplement are generated from registered result files; absent artifacts are reported rather than estimated.",
            r"\paragraph{Protocol and provenance.} The artifact manifest records each experiment directory, iteration, and source JSON. Per-view comparisons use only matched view identifiers.",
            r"\input{supplementary_experiments/tables/main_metrics.tex}",
            r"\paragraph{Per-view statistics.} Bootstrap confidence intervals and win rates are available only for matched main-method and comparator views. The corresponding CSV preserves view-level sources.",
            r"\input{supplementary_experiments/tables/efficiency.tex}",
            r"\paragraph{Convergence and diagnostics.} Training-log PSNR and anchor-event plots, plus render diagnostics, are included only when their logged artifacts are available.",
            r"\paragraph{Limitations.} This supplement does not claim evidence for unavailable ASG-vs-SG, prior-splitting, seed-robustness, or cross-view-warping comparisons.",
            "",
        ]
    )
    (report_dir / "supplementary_experiments.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TOOL_ROOT / "configs" / "experiment_manifest.yaml")
    parser.add_argument("--outputs-root", type=Path, help="Override manifest outputs_root; paths may be absolute.")
    parser.add_argument("--no-plots", action="store_true", help="Build data, tables, and report without importing Matplotlib.")
    args = parser.parse_args()
    global PLOTS_DISABLED
    PLOTS_DISABLED = args.no_plots
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    configured_root = args.outputs_root or resolve_path(manifest.get("outputs_root"), REPO_ROOT) or DEFAULT_OUTPUTS_ROOT
    outputs_root = configured_root.resolve()
    data_dir, figures_dir, tables_dir, report_dir = (TOOL_ROOT / name for name in ("data", "figures", "tables", "report"))
    for directory in (data_dir, figures_dir, tables_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    completeness: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    per_view_rows: list[dict[str, Any]] = []
    curves_by_id: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen_ids: set[str] = set()
    for raw_entry in manifest["experiments"]:
        if not isinstance(raw_entry, dict) or not raw_entry.get("enabled", True):
            continue
        if not raw_entry.get("id") or not raw_entry.get("scene"):
            raise ValueError("Every enabled manifest experiment needs non-empty 'id' and 'scene'.")
        experiment_id = str(raw_entry["id"])
        if experiment_id in seen_ids:
            raise ValueError(f"Duplicate experiment id: {experiment_id}")
        seen_ids.add(experiment_id)
        record, rows, curves = collect_experiment(raw_entry, outputs_root, completeness)
        records.append(record)
        per_view_rows.extend(rows)
        curves_by_id[experiment_id] = curves

    summary_rows = metric_summary_rows(records)
    statistics = paired_statistics(per_view_rows, int(manifest.get("bootstrap_samples", 2000)))
    write_json(data_dir / "experiment_records.json", {"manifest": str(manifest_path), "outputs_root": str(outputs_root), "records": records})
    write_json(data_dir / "completeness_report.json", {"missing_or_unreadable": completeness, "count": len(completeness)})
    write_csv(data_dir / "metric_summary.csv", summary_rows, ["experiment_id", "scene", "method", "role", "report", "iteration", "PSNR", "SSIM", "LPIPS", "source"])
    write_csv(data_dir / "per_view_metrics.csv", per_view_rows, ["experiment_id", "scene", "display_name", "role", "report", "view", "PSNR", "SSIM", "LPIPS", "source"])
    write_csv(data_dir / "per_view_statistics.csv", statistics, ["scene", "report", "reference", "comparator", "metric", "matched_views", "mean_delta_reference_minus_comparator", "bootstrap_ci95_low", "bootstrap_ci95_high", "reference_win_rate", "tie_rate"])

    figures = []
    figures.extend(write_summary_plots(summary_rows, figures_dir))
    figures.extend(write_per_view_plots(per_view_rows, statistics, figures_dir))
    efficiency_rows, efficiency_figures = write_efficiency_table_and_plot(records, tables_dir, figures_dir)
    figures.extend(efficiency_figures)
    figures.extend(write_convergence_plots(curves_by_id, records, figures_dir, data_dir))
    figures.extend(write_qualitative_figures(records, manifest, figures_dir))
    write_latex_tables(summary_rows, efficiency_rows, tables_dir)
    if PLOT_ERROR:
        completeness.append({"experiment_id": "toolkit", "artifact": "Matplotlib plotting backend", "path": sys.executable, "status": f"unavailable: {PLOT_ERROR}"})
        write_json(data_dir / "completeness_report.json", {"missing_or_unreadable": completeness, "count": len(completeness)})
    write_report(records, completeness, figures, summary_rows, statistics, report_dir)
    print(f"Supplementary evidence build complete: {TOOL_ROOT}")
    print(f"Registered experiments: {len(records)} | missing/unreadable artifacts: {len(completeness)} | generated figures: {len(figures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
