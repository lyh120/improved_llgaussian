# LLNeRF needle artifact investigation

Use `run_llnerf.py` for scenes under `/home/liuyuhao/datasets/llnerf-dataset`.
It prepares a workspace copy, gives every run its own CIDNet cache, and trains
with a bound on the final Gaussian axis ratio.

```bash
# Indoor scene: verified cleanest setting for room
python scripts/run_llnerf.py room --gpu 1 --run-tag clean

# Fine vegetation: more shape freedom
python scripts/run_llnerf.py shrub --gpu 1 --run-tag cap8 \
  --max-gaussian-anisotropy 8 --reflectance-texture-scale 1

# Adjust the two CIDNet controls; use a fresh tag for each choice
python scripts/run_llnerf.py book --gpu 1 --run-tag gamma08_alpha10 \
  --cidnet-gamma 0.8 --cidnet-alpha 1.0

# Follow the bike command's 100-step CIDNet controller fit, starting at 0.9/0.9
python scripts/run_llnerf.py room --gpu 1 --run-tag cidnet_fit100 \
  --cidnet-gamma 0.9 --cidnet-alpha 0.9 --cidnet-mlp-steps 100

# Use selected 2D targets. Files must match the source image stems and dimensions.
python scripts/run_llnerf.py book --gpu 1 --run-tag selected \
  --enhanced-dir /path/to/enhanced/images
# Add --enhanced-already-undistorted if those targets already use pinhole geometry.
```

The source data is left untouched. `datasets/llnerf_prepared/<scene>` contains
undistorted images, a pinhole camera file, and `preparation.json` with a voxel
size derived from the median nearest point spacing. The copied `images.bin`
retains its original 2D feature observations; this copy is for this trainer.
Run inputs and CIDNet targets live separately under
`datasets/llnerf_prepared/_runs/<scene>_<tag>`.

| Observation | shrub | room |
| --- | ---: | ---: |
| Radial coefficient | -0.07057 | 0.000296 |
| Approximate image corner distortion | 35 px | 0.14 px |
| Median sparse point spacing | 0.21557 | 0.13498 |
| Selected voxel size | 0.01347 | 0.00844 |
| Selected final Gaussian axis ratio limit | 8 | 4 |

The original bike setting used `--voxel_size 0.0005`. In shrub, the unrestricted
6000-step model had 651 stored anchors with axis ratio above 20 and a maximum
ratio near 2816. Long lines were visible in its reflectance render. Bounding
the *final* three Gaussian scales, including the covariance MLP output,
suppressed those lines. Retraining restored coverage lost when the old
checkpoint was merely rerendered with the new bound.

On the three shrub test reflectance images, a Hough diagnostic counted 488
segments of at least 250 px without the shape limit and 212 with limit 8.
Low-light test PSNR was 47.2965 versus 47.3020. On four room test reflectance
images, the count of segments of at least 150 px fell from 198 with limit 8
to 66 with limit 4. Setting `--reflectance-texture-scale 0` with limit 4
reduced it further to 17. The three scaled losses amplify local contrast in
very dark source images and promoted short scratches. The texture-free run
retained the object outlines but softened some fine label detail. Test PSNR
was 32.8969, 32.8983, and 32.8619 respectively. The runner defaults to
texture scale 0 for LLNeRF; set 1 to recover the bike losses, especially for
fine vegetation. Hough counts include real straight edges, so inspect the
exported images directly.

CIDNet preview on an extremely dark shrub image favored the conservative
`gamma=0.9, alpha=0.9` pair. Gamma 0.6 increased visible color noise; alpha
1.4 nearly clipped highlights. The runner fixes the CIDNet MLP at zero steps
by default so these two controls have repeatable effects. Override
`--cidnet-mlp-steps` if per-image fitting is desired. With 100 fitting steps,
`--cidnet-gamma` and `--cidnet-alpha` are initial values, not fixed output
values: the existing bike cache initialized at 0.9/0.9 and finished with
mean gamma/alpha near 0.949/0.984. The verified LLNeRF runs used zero steps
and exactly 0.9/0.9, so the 100-step command above is an unverified option.
Selected 2D targets override generated CIDNet targets for that run.

The enhanced renders still show color and noise errors from very dark source
frames. The PSNR above measures reconstruction of the low-light images and
does not measure enhanced image quality. Verified 6000-step outputs are in
`experiments/llnerf/shrub_cap8` and `experiments/llnerf/room_cap4_tex0`.

## Extended training and component experiments

The following experiments were run after the initial 6000-step comparison.
The shrub metrics use test reflectance Hough segments of at least 250 px;
room uses 150 px. The Laplacian number is the mean variance over test
enhanced renders and is only a relative sharpness diagnostic.

| Experiment | Test sharpness | Long lines | Low-light PSNR | Result |
| --- | ---: | ---: | ---: | --- |
| shrub_detail12k_clean2 | 27.84 | 10 | 47.318 | clean but still soft |
| shrub_detail12k_edge | 50.02 | 64 | 47.296 | sharpness comes with scratches |
| shrub_detail20k_dense | 49.77 | 41 | 47.329 | best detail, residual needles |
| shrub_rl_color12k | 30.73 | 10 | 47.258 | fewer needles, color still unstable |
| room_detail12k_edge03 | 18.02 | 25 | 32.845 | small improvement over 6k |
| room_rdetail12k_dense | 13.78 | 17 | 32.855 | R detail loss is stable, gain is small |

`room_rdetail_only6k` adds the geometry-detached local detail loss to the
reflectance raster. It improved test sharpness from 12.31 to 13.35 and reduced
long lines from 18 to 12. This loss is exposed as
`--reflectance-target-detail-reg`; it compares local log-luminance detail in R
to the selected CIDNet target while keeping Gaussian geometry detached.

The component implementation is in `utils/enhancement_loss_utils.py`, with the
detached reflectance raster built in `gaussian_renderer/__init__.py`. The
runner also supports selected targets, denoising, direct bright-input trials,
and the two CIDNet controls. Use a fresh `--run-tag` for every run.

Recommended shrub command (best tested detail, use axis ratio 4 at render time
to reduce residual needles):

```bash
python scripts/run_llnerf.py shrub --gpu 1 --run-tag detail20k_dense_repro \
  --high-detail --iterations 20000 --max-anchors 180000 --update-until 16000 \
  --max-gaussian-anisotropy 8 --enhancement-reflectance-reg 1.0 \
  --enhancement-guidance-final-weight 1.0 --enhancement-degree-reg 0 \
  --enhancement-degree-global-reg 0 --cidnet-gamma 0.9 --cidnet-alpha 0.9
```

For a cleaner shrub result, render the same checkpoint with
`--max_gaussian_anisotropy 4`; the tested 20k checkpoint went from 33 to 25
long lines with essentially unchanged sharpness. For room, use the adaptive
initial reflectance and depth settings supplied by the runner and
`--reflectance-target-detail-reg 0.6`; larger color regularizers were rejected
because they reduced sharpness and PSNR.

Generated 2D targets can be denoised reproducibly with:

```bash
python scripts/denoise_llnerf_targets.py SOURCE_DIR DEST_DIR --strength 8
```

The target denoising reduced room long lines from 25 to 18 but also reduced
sharpness from 18.02 to 12.31, so it is useful for artifact suppression rather
than the final clear reconstruction. The direct R*L raster path and projected
sigma cap were tested and left disabled because they reduced detail or coverage.

Background: [COLMAP camera models](https://colmap.github.io/cameras.html) and
[Spectral-GS](https://arxiv.org/abs/2409.12771).
