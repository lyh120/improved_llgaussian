# Supplementary experiment evidence toolkit

This folder turns existing LL-Gaussian experiment artifacts into traceable figures,
tables, and a supplement draft. It **never trains a model** and it never invents a
missing metric. Every number in generated tables includes its source path in
`data/experiment_records.json`.

## Quick start

1. Edit `configs/experiment_manifest.yaml` and add one item for every baseline,
   main method, or ablation. Directory names are deliberately unconstrained.
2. On the machine that can read the experiment outputs, run:

   ```bash
   python supplementary_experiments/build.py \
     --manifest supplementary_experiments/configs/experiment_manifest.yaml \
     --outputs-root /home/liuyuhao/ll_further/LL-Gaussian-sg/outputs
   ```

3. Collect the generated artifacts from `data/`, `figures/`, `tables/`, and
   `report/`. These generated files are ignored by Git so they do not alter raw
   experiment outputs or source control state.

The default manifest already registers the five supplied primary runs (buu, sofa,
bike, chair, and shrub). It assumes they live below `outputs/`; use
`--outputs-root` on the Linux training machine when that directory is elsewhere.

## Manifest fields

The manifest is JSON-formatted YAML, so it works without adding a new dependency.
It can also be rewritten as ordinary YAML when `PyYAML` is installed.

```yaml
outputs_root: /home/liuyuhao/ll_further/LL-Gaussian-sg/outputs
bootstrap_samples: 2000
qualitative_views:
  bike: ["7"]
experiments:
  - id: bike_alis
    scene: bike
    display_name: ALIS-GS
    role: main                 # main, baseline, ablation, diagnostic
    output_dir: bike_asg_nor_8k_detail
    iteration: 8000            # integer, latest, or -1
    enabled: true
    peak_memory_gb: 8.4        # optional manual timing/memory field
    log_path: /path/to/train.log # optional; absolute or repo-relative
```

`output_dir` is relative to `outputs_root`, unless it is absolute. The tool reads
the standard `test/ours_<iteration>/` layout and accepts `result_dir` when a run
uses a non-standard result location. It reads the following files when present:

- `metrics_enhanced_gt.json` and `metrics_lowlight.json`
- `render_timing_profile.json`
- `coverage_stats.json` and `illumination_stats.json` / `asg_stats.json`
- `point_cloud/iteration_<N>/point_cloud.ply`
- render directories (`renders`, `render_enhanceds`, `gt`, and diagnostics)
- the configured training log

## Outputs and evidence policy

- `data/completeness_report.json` lists every missing expected artifact.
- `data/per_view_statistics.csv` contains paired win rates and deterministic
  bootstrap confidence intervals only when two methods share view names.
- Figures and tables omit unsupported comparisons instead of filling cells with
  zeros. The report uses `—` for unavailable values and records why.
- `report/supplementary_experiments.tex` is an evidence-bound draft; edit its
  prose after reviewing the generated figures, but do not turn absent comparisons
  into claims.
