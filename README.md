# LL-Gaussian Explicit R/L v2

This branch keeps the Scaffold-GS anchor/offset geometry while replacing every appearance network with explicit parameters. `MODEL_FORMAT_VERSION=2` is an intentional format break: checkpoints and PLY files from earlier branches are rejected.

## Model boundary

The only remaining MLPs are the Scaffold-GS geometry heads:

- `mlp_opacity`
- `mlp_cov`
- optional `mlp_feature_bank`

Appearance is owned by `ExplicitAppearance` and contains no `Linear`, decoder, embedding, residual, noise, artifact, or runtime mode switch.

For anchor `a` and offset `k`, reflectance is

```text
centered_detail[a,k] = detail[a,k] - mean_k(detail[a,k])
R[a,k] = sigmoid(base[a] + centered_detail[a,k])
```

Main illumination is a single grayscale ASG with explicit axis, tangent, two-axis sharpness, energy, and ambient. Enhanced appearance combines bounded RGB diffuse illumination with an independent additive single RGB SG. Directions are normalized, ASG tangents are orthogonalized, and sharpness is `softplus(raw) + NUMERICAL_EPS`.

The enhanced branch is bounded by construction:

```text
D           = sigmoid(diffuse_raw)
S(v)        = sigmoid(energy_raw) * exp(softplus(sharpness_raw) * (dot(v, axis) - 1))
B_enhanced  = stop_gradient(R) * D
C_enhanced  = B_enhanced + (1 - B_enhanced) * S(v)
L_enhanced  = D + (1 - D) * S(v)              # diagnostic response for R=1
```

Every enhanced quantity is in `[0,1]` without an output clamp. Initialization
analytically splits the prior into the part no brighter than R and a remaining
headroom fill; it never evaluates `enhancement_prior / R`. Early v2 artifacts
with the unstable unbounded-gain layout are rejected by the explicit appearance
layout marker rather than silently reinterpreted.

The renderer composes colors in Gaussian space:

```text
low color       = R * L
enhanced color  = B_enhanced + (1 - B_enhanced) * S(v)
```

Enhanced rendering detaches position, opacity, covariance, scale, and rotation, so its loss updates only the enhanced SG. Reflectance and illumination diagnostics also use detached Scaffold geometry. The diagnostic fields and W&B images are coverage-conditioned means rather than black-background, alpha-premultiplied accumulations:

```text
coverage       = sum_i w_i
attribute_sum  = sum_i w_i * X_i
render_X       = attribute_sum / (coverage + NUMERICAL_EPS)
X in {R, L, L_enhanced}
```

Diagnostic geometry and coverage are detached, so this normalization does not alter gradient ownership. The low-light image is still rasterized once from Gaussian-space `R * L`; in general, the two diagnostic means are not expected to multiply back into that image when Gaussians overlap. `render_depth` is the geometry-connected expected depth `(sum_i w_i z_i) / (sum_i w_i + NUMERICAL_EPS)`, not alpha-premultiplied accumulated depth.

## Installation

```bash
conda create -n llgaussian python=3.10
conda activate llgaussian
pip install -r requirements.txt
```

The residual rasterizer is not a dependency of v2. StableSR is an optional,
offline enhancement-prior backend and never enters the training graph.

## Scene-local fixed priors

By default, `train.py -s DATASET/SCENE` stores fixed priors inside that scene.
The enhancement directory depends on `--enhancement_prior_backend`:

```text
CIDNet:   DATASET/SCENE/cidnet_prior
StableSR: DATASET/SCENE/diffusion_prior_<stablesr_input_gain>
Depth:    DATASET/SCENE/depth_maps
```

At startup, a cache with `manifest.json` is reused. A missing or empty cache is
generated once by the standalone preprocessors before optimization begins. A
nonempty directory without a v2 manifest is rejected and never overwritten.
The training loop itself never refreshes or reruns either prior model. Explicit
`--enhancement_prior_path` and `--depth_prior_path` arguments override these
scene-local defaults.

Startup is intentionally ordered and visible: DepthAnything is prepared or
reused first, the selected enhancement backend second, then both manifests are
loaded with per-image progress bars. Model inference runs only for a missing or
empty cache; a valid manifest always takes the reuse path.

`--enhancement_prior_backend` accepts `cidnet` (default) or `stablesr`. Both
produce the same strict `enhancement_rgb` interface, so the renderer and loss
are backend-independent. StableSR follows the clean-project prior recipe:
multiply the low-light input by an explicit `--stablesr_input_gain`, run the
fixed StableSR-Turbo prior once, and supervise `render_enhanced` with the same
photo loss used by CIDNet. No degree, color, smoothness, refresh, or controller
loss is restored. The gain defaults to the explicit clean-recipe value `15.0`;
override it explicitly for another recipe instead of deriving it from a hidden
target-mean constant.
StableSR code and config are resolved below `submodules/StableSR`, while
`stablesr_turbo.ckpt` and `vqgan_cfw_00011.ckpt` are resolved from the
project-level `checkpoints/` directory, matching the clean-project layout.

The same preprocessing can still be run manually. `--eval`, `--lod`, and
`--llffhold` must match training so the manifest contains exactly the training
`image_name` set.

```bash
python scripts/precompute_cidnet_prior.py \
  --source_path ./dataset/LLRS-sRGB/chair \
  --output ./dataset/LLRS-sRGB/chair/cidnet_prior \
  --cidnet_root ./submodules/HVI-CIDNet \
  --weights ./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth \
  --eval

python scripts/precompute_stablesr_prior.py \
  --source_path ./dataset/LLRS-sRGB/chair \
  --output ./dataset/LLRS-sRGB/chair/diffusion_prior_15 \
  --stablesr_root ./submodules/StableSR \
  --inference_script ./submodules/StableSR/scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
  --config ./submodules/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
  --checkpoint ./checkpoints/stablesr_turbo.ckpt \
  --vqgan_checkpoint ./checkpoints/vqgan_cfw_00011.ckpt \
  --input_gain 15 --steps 4 --decoder_weight 0.75 --colorfix_type wavelet \
  --eval

python scripts/precompute_depth_prior.py \
  --source_path ./dataset/LLRS-sRGB/chair \
  --output ./dataset/LLRS-sRGB/chair/depth_maps \
  --depth_anything_root ./submodules/Depth-Anything-V2 \
  --encoder vitl \
  --checkpoint ./checkpoints/depth_anything_v2_vitl.pth \
  --eval
```

The enhancement cache contains lossless PNGs. The depth cache contains float32 relative-disparity `.npy` files. Both have a strict v2 `manifest.json`; missing, duplicate, extra, non-finite, wrong-size, or wrongly named entries fail immediately.

## Training

```bash
bash scripts/train.sh \
  --data ./dataset/LLRS-sRGB/chair \
  --model ./outputs/chair_explicit_v2 \
  --gpu 0 --iterations 8000 --warmup --wandb
```

The wrapper explicitly uses `--prune_ratio 1.0`, which retains the complete input point cloud, and stops main densification at three quarters of the requested iterations. The remaining quarter is a refinement tail for inherited explicit appearance parameters. Its 8k schedule uses a 30k anchor ceiling, at most 256 new anchors per main update, and at most 128 per warmup update; training-time pruning is disabled for the first diagnosis run. `prune_ratio` controls only the initial random point-cloud reduction; it is unrelated to training-time anchor pruning. A direct `train.py` invocation that has a nonzero growth budget must set `update_until < iterations`, otherwise startup fails instead of silently saving immediately after the last growth event. Set `max_new_anchors_per_update=0` for a no-growth smoke test. Training-time pruning is enabled only when `max_pruned_anchors_per_update` is positive.

To select StableSR instead, add:

```bash
--enhancement-backend stablesr --stablesr-input-gain 15
```

If StableSR uses a separate compatible environment, pass its interpreter to
`train.py` with `--stablesr_python /path/to/stablesr-env/bin/python`.

When W&B is enabled, the optional `--wandb_monitor_camera` (an exact
`image_name` or zero-based sorted index),
`--wandb_monitor_split {train,test}`, and `--wandb_monitor_interval` arguments
select one fixed monitoring view. Only `image`, `image_enhanced`, `reflectance`,
`illumination`, and `illumination_enhanced` are logged as images; scalar loss and
densification logs remain available. As in the clean project, only the dark
`image` and grayscale `illumination` views receive a scene-level display gain;
the gain is logged and never enters priors, rendered training targets, or loss.

Warmup uses the same appearance computation and the same loss as main training. It only scales Scaffold geometry learning rates and changes the anchor densification window; the pose and appearance schedules remain unchanged.
`warmup_geometry_lr_scale=0.1` is a named Scaffold-only multiplier; it never changes the shared explicit appearance LR. At the warmup-to-main boundary, only the interval-local offset gradient numerator/denominator are reset. Lifetime opacity, visibility, anchor birth, and optimizer state are preserved. W&B records the phase, the real learning rate of every optimizer group, statistics-collection/adjust/reset events, and the enhanced-illumination mean/p95/max without adding any images beyond the five core views.

Checkpoints store the warmup enable flag, boundary, and whether the transition
has completed. Resume must use the original `--warmup` and
`--warmup_iterations` values; a mismatch fails before optimizer restoration.
Thus a checkpoint saved at iteration 2000 resets the interval-local evidence
once on iteration 2001, while a post-transition checkpoint never resets twice.
The terminal uses separate `Training [warmup]` and `Training [main]` progress
bars and prints a framed `[phase 2/2]` transition, so the 2k+8k split is not
hidden inside a single ambiguous counter.

Max-RGB multi-view initialization also performs a deterministic per-pixel depth
test: among projected anchor-offset samples that round to the same pixel, only
the nearest sample contributes. Unobserved offsets still use global robust
statistics; hidden samples are no longer mixed into their R/L estimate.

The complete objective is

```text
L = L_photo_low
  + lambda_reflectance_reconstruction * L_photo(R * stopgrad(L), I_low)
  + lambda_illumination * (L_photo(L, MaxRGB(I_low)) + L_edgeTV)
  + lambda_enhanced * (L_photo_enhanced + L_photo(L_enhanced, H_target))
  + lambda_depth                * L_depth
  + lambda_scaling              * L_scale
```

Only six loss arguments are public:

| Argument | Default |
| --- | ---: |
| `lambda_dssim` | `0.2` |
| `lambda_scaling` | `0.01` |
| `lambda_reflectance_reconstruction` | `1.0` |
| `lambda_illumination` | `1.0` |
| `lambda_enhanced` | `1.0` |
| `lambda_depth` | `1.0` |

All explicit appearance tensors share one schedule:

| Argument | Default |
| --- | ---: |
| `explicit_appearance_lr_init` | `0.008` |
| `explicit_appearance_lr_final` | `0.00005` |

`NUMERICAL_EPS=1e-6` is used only for division, normalization, finite-value protection, and positive sharpness. Scaffold-GS geometry constants such as the visibility margin and filter footprint are separately named and documented in the geometry module.

## Rendering and metrics

```bash
python render.py \
  -m ./outputs/chair_explicit_v2 \
  --dataset_path ./dataset/LLRS-sRGB/chair \
  --iteration 8000 --skip_train --profile_render_timing
```

The renderer writes:

- `renders`
- `render_enhanceds`
- `render_reflectances`
- `render_illuminations`
- `render_illuminations_enhanced`
- `render_depths`
- `metrics_lowlight.json`
- `metrics_enhanced_gt.json` when enhanced GT is available
- `profile.json` with FPS, parameter count, and serialized model size when profiling is enabled

The three R/L diagnostic directories contain coverage-conditioned covered means. `render_depths` contains expected depth; neither output is a raw alpha-premultiplied raster attribute.

## Format v2

- Training checkpoints are named dictionaries with model version, iteration, warmup-stage metadata, Scaffold tensors, explicit appearance, optimizer state, and densification state.
- PLY stores a mandatory version property and the appearance-format-v2 bounded diffuse/additive-SG layout.
- `save_mlp_checkpoints` and `load_mlp_checkpoints` serialize only opacity/covariance/optional feature-bank heads.
- There is no legacy backfill or format guessing. Start a new v2 experiment.

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile train.py render.py scene/explicit_appearance.py scene/gaussian_model_v2.py gaussian_renderer/__init__.py
```

CPU tests cover explicit R/ASG/SG behavior, finite gradients, robust initialization, coverage-conditioned losses, densification schedule rejection, grow/prune alignment, strict priors, AST boundaries, and v2 PLY/checkpoint round trips. CUDA tests additionally cover opacity-invariant normalized attributes/expected depth and rasterizer gradient isolation; train/render smoke tests run only in a configured CUDA environment.

## License and upstream project

This repository derives from LL-Gaussian and Scaffold-GS. Follow the licenses in [LICENSE.md](LICENSE.md) and the bundled submodules.
