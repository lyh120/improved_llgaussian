# Pure explicit R/L mode

Use `--pure_explicit_rl --reflectance_mode explicit --illumination_mode asg`.
Legacy R/L checkpoints remain loadable when this flag is absent. Scaffold-GS
opacity and covariance MLPs remain active for Gaussian geometry.

For anchor `i` and offset `k`, reflectance is stored directly:

```
R_ik = clamp(exp(B0_i + tanh(delta_ik)), eps, 1)
```

`eps` is only a numerical positive floor. With
`--explicit_feature_conditioning`, Scaffold-GS features provide a bounded
local modulation while the stored explicit value remains the material base:

```
R'_ik = clamp(R_ik * exp(tanh(g_theta([f_i, o_ik]))), eps, 1)
```

The baseline illumination is the explicit ASG in `utils/sg_utils.py`, with
per-offset axis `z`, orthogonal tangent axes `x,y`, two sharpness values,
amplitude and bias. For view direction `v` and default one lobe:

```
lambda_x/y = softplus(raw_lambda_x/y) + lambda_min
L_ik(v) = clamp(sigmoid(b_ik) + sigmoid(a_ik) * max(0, dot(v,z_ik))
           * exp(-lambda_x * dot(v,x_ik)^2 - lambda_y * dot(v,y_ik)^2), 0, 1)
```

Multiple lobes sum inside the clamp; optional distance falloff multiplies
their sum. The legacy pure mode uses an explicit per-offset colored SG
increment:

```
L_enh_ik(v) = clamp(L_ik(v) + sigmoid(a_enh_ik)
                     * exp(softplus(lambda_enh_ik) * (dot(z_enh_ik, v) - 1)), 0, 1)
```

The feature-conditioned mode uses the original Scaffold-GS-style enhancement
network instead:

```
L_enh = MLP_theta([stopgrad(f_i), stopgrad(L_i)])
```

Its sigmoid output is bounded. The explicit ASG remains the base illumination
and old pure-explicit checkpoints remain readable.

The pure mode bypasses the reflectance decoder. The feature-conditioned mode
replaces it with the bounded feature conditioner above and restores the
original feature-conditioned enhancement MLP. Both modes keep primary color
as a per-Gaussian product rasterized once: `Rasterize(R_ik * L_ik)`.

For a 2D enhancement target `T`, the illumination and reflectance guidance
branches use the same single-pass image formation:

```
E = Rasterize(R_ik * L_enh_ik)
loss_illum = |Rasterize(stopgrad(R_ik) * L_enh_ik) - T|_1
loss_refl  = |Rasterize(R_ik * stopgrad(L_enh_ik)) - T|_1
```

The latter branch also detaches positions, opacities, covariance, camera pose,
and projected means from the 2D enhancement target. Low-light reconstruction
still trains geometry through the ordinary Scaffold-GS branch. This matters
when the teacher contains frame-specific noise that would otherwise create
long, thin Gaussian artifacts.

## LOM full dataset

Run `python scripts/run_lom_pure_explicit.py`. It covers the five prepared
standard scenes: bike, buu, chair, shrub and sofa. Each uses the successful
6000-step LOM strategy, with the pure explicit flag added. Results are saved
under `experiments/lom_pure_explicit/SCENE`.

| Scene | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| bike | 47.02543 | 0.98491 | 0.06110 |
| buu | 44.20124 | 0.98219 | 0.07472 |
| chair | 47.49369 | 0.98810 | 0.09976 |
| shrub | 38.75889 | 0.90032 | 0.21502 |
| sofa | 46.27638 | 0.98557 | 0.13667 |

The feature-conditioned sofa validation was run with the same 6000-step
recipe and `--explicit_feature_conditioning`. It reached 46.14491 PSNR,
0.98582 SSIM and 0.13282 LPIPS. The baseline remains slightly better in PSNR
but the conditioned model improves SSIM/LPIPS, so the added feature coupling
is compatible and useful without replacing the explicit base.

These are test metrics against the low-light input frames, not bright
reference metrics. The `buu` legacy baseline is 44.01385/0.98394/0.07208.

## LLNeRF flower

`flower` was prepared from `/home/liuyuhao/datasets/llnerf-dataset` as a new
scene, separate from the earlier shrub and room experiments. It has 17 train
and 3 test views. The following values come from `render.py --skip_optimize`
on the three held-out low-light images. The Laplacian value is a rough
enhanced-image edge proxy, not a reference-based sharpness metric.

| Run | Steps | PSNR | SSIM | LPIPS | Enhanced Laplacian variance | Long Hough lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw CIDNet target | 6000 | 39.9121 | 0.90015 | 0.29852 | 3.95 | 0 |
| raw CIDNet target | 12000 | 39.9117 | 0.90021 | 0.29715 | 5.15 | 0 |
| denoised, balanced target | 6000 | 39.9169 | 0.90005 | 0.30024 | 3.13 | 0 |
| balanced target + explicit R guidance | 6000 | 39.9230 | 0.90027 | 0.29992 | 3.01 | 0 |
| preceding checkpoint, Gaussian render scale 0.85 | 6000 | 39.6309 | 0.89478 | 0.28539 | 4.37 | 0 |

The balanced target corrects part of the strong purple cast, and the
single-pass R guidance improves the low-light reconstruction slightly. It
does **not** recover crisp enhanced detail: the teacher frames are soft, and
the three held-out enhanced views lack bright ground truth. The 0.85 render
scale trades PSNR for better LPIPS and visible edges; it is a rendering
choice, not a training fix. The long-line detector used Canny 40/100 and
HoughLinesP with a 250-pixel minimum length on each of the three enhanced
test renders. Zero detected lines is limited evidence about needle artifacts,
not a guarantee for every view.

Reproduce the strongest color-corrected flower checkpoint:

```
python scripts/run_llnerf.py flower --gpu 1 --port 45733 \
  --run-tag pure_refined_r02_6k --pure-explicit-rl --high-detail \
  --iterations 6000 --max-anchors 90000 --update-until 5500 \
  --max-gaussian-anisotropy 4 --reflectance-target-detail-reg 0.3 \
  --reflectance-target-chroma-reg 0.2 \
  --enhancement-illumination-chroma-reg 0.05 \
  --enhancement-reflectance-reg 0.2 \
  --enhanced-dir datasets/llnerf_targets/flower_cidnet_refined \
  --enhanced-already-undistorted --cidnet-mlp-steps 0

CUDA_VISIBLE_DEVICES=1 python render.py \
  -m experiments/llnerf/flower_pure_refined_r02_6k \
  --iteration 6000 --skip_train --skip_optimize \
  --gaussian_render_scale 0.85
```
