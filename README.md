<h1 align='center'>LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis</h1>
<div align='center'>
    <a href='https://github.com/sunhao242/LL-Gaussian' target='_blank'>Hao Sun</a><sup>1,2</sup> 
    <a href='https://fenggenyu.github.io/' target='_blank'>Fenggen Yu</a><sup>4</sup> 
    <a href='https://github.com/sunhao242/LL-Gaussian' target='_blank'>Huiyao Xu</a><sup>3</sup> 
    <a href='https://github.com/sunhao242/LL-Gaussian' target='_blank'>Tao Zhang</a><sup>5</sup> 
    <a href='https://changqingzou.weebly.com/' target='_blank'>Changqing Zou†</a><sup>1,3</sup> 
</div>

<div align='center'>
    <sup>1</sup>Zhejiang Lab  <sup>2</sup>University of Chinese Academy of Sciences  <sup>3</sup>State Key Lab of CAD&CG, Zhejiang University <sup>4</sup>Simon Fraser University <sup>5</sup>Hangzhou Dianzi University
</div>
<div align='center'>
    †Corresponding Author
</div>
<div align="center">
    <strong>ACM Multimedia 2025</strong>
</div>
<br>
<div align="center">

[![Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Demo-00bfff)](https://sunhao242.github.io/LL-Gaussian_web.github.io/)
[![Paper](https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=acm)](https://dl.acm.org/doi/pdf/10.1145/3746027.3755375)
[![Arxiv](https://img.shields.io/static/v1?label=Arxiv&message=PDF&color=red&logo=arxiv)](https://dl.acm.org/doi/pdf/10.1145/3746027.3755375)
[![Dataset](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Dataset&message=GoogleDrive&color=green)](https://img.shields.io/static/v1?label=Dataset&message=GoogleDrive&color=yellow&logo=google)


</div>
<p align="center">
  <img src="assets/teaser.png"  height=400>
</p>

-----


## 🔥 News

+ [2026.04.19] 📢 Training code is released.
+ [2026.04.13] 🔭 Inference code is released.
+ [2025.10.22] 🎆 LLRS dataset is released.
+ [2025.07.06] 🎉 LL-Gaussian is accepted by ACM Multimedia 2025!
+ [2025.04.19] 🚀 Repo is created!

🤗 If you find LL-Gaussian useful for your projects, a ⭐ would be greatly appreciated. Thanks! 🤗



## 📌 TODO

- [x] Release LLRS dataset.
- [x] Release inference code.
- [x] Release training code.

## ✨ Key Features
- 🌙 **Low-Light Gaussian Initialization**: Robust 3D Gaussian initialization without reliance on SfM tools like COLMAP.
- 🔀 **Gaussian Decomposition**: Dual-branch modeling of intrinsic vs. transient scene components.
- 🎥 **Novel View Synthesis in the Dark**: State-of-the-art rendering quality under low-light and nighttime conditions.


---

## 📦 Installation


### 1. Clone repository

```bash
git clone --recursive https://github.com/sunhao242/LL-Gaussian.git 
cd LL-Gaussian
```

### 2. Create environment

```bash
conda create -n llgaussian python==3.10
conda activate llgaussian
pip install -r requirements.txt
```

## 📊 Dataset

- Download the LLRS-sRGB dataset and place it under:

```bash
./dataset/LLRS-sRGB
```

👉 Download from [Link](https://drive.google.com/file/d/1Y5lhAEXFN0lZDN-ITPPVtjm42-jKk9JR/view?usp=sharing).



## ⚡ Quick Inference 

### 1. Download pretrained checkpoint

Download checkpoint from [backup](https://drive.google.com/file/d/1Mf7pG5Lm5N3ybfpgNuvy9PMQgrdgaMVN/view?usp=drive_link).
Place it under:

```bash
./backup
```

### 2. Run inference


```bash
python render.py -m ./backup/LLRS-sRGB/{scene_name}/{XXXX-XX-XX_XX:XX:XX} --dataset_path ./dataset/LLRS-sRGB/{scene_name} --skip_train
```


## 🏋️ Training

Step 1: Download StableSR checkpoints

```bash
./checkpoints/
  ├── vqgan_cfw_00011.ckpt
  ├── stablesr_turbo.ckpt
```

  - Download autoencoder from [Huggingface](https://huggingface.co/Iceclear/StableSR/resolve/main/vqgan_cfw_00011.ckpt) 
  - Download StableSR-Turbo from [Huggingface](https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_turbo.ckpt) 

Step 2: Download the Depth Anything V2 

Download [Depth-Anything-V2-Large](Depth-Anything-V2-Large) to:

```bash
./checkpoints/
```

Step 3: Start training

```bash
bash scripts/single_train.sh
```

### Reproducible Best Setting (LLRS-sRGB/chair, 8k)

The following command is the current best-performing setting in this repo for
`LLRS-sRGB/chair` under the dual-transient pipeline:

```bash
python train.py --eval \
  -s /home/liuyuhao/ll_further/LL-Gaussian/dataset/LLRS-sRGB/chair \
  -m outputs/chair_sg_dualtransient_v8 \
  --gpu 1 \
  --use_sg_illumination --illumination_mode sg \
  --use_3D_filter --use_residual --use_wandb --warmup \
  --use_dual_transient \
  --iterations 8000 \
  --save_iterations 8000 \
  --test_iterations 5000 8000 \
  --position_lr_max_steps 8000 \
  --offset_lr_max_steps 8000 \
  --update_from 1500 \
  --update_until 4500 \
  --start_stat 500 \
  --mlp_opacity_lr_max_steps 8000 \
  --mlp_cov_lr_max_steps 8000 \
  --mlp_color_lr_max_steps 8000 \
  --mlp_color_lr_init 0.008 \
  --mlp_color_lr_final 0.00025 \
  --offset_lr_init 0.001 \
  --offset_lr_final 0.00001 \
  --feat_dim 32 \
  --reflectance_consistency_reg 2e-5 \
  --reflectance_smooth_reg 0.0 \
  --reflectance_edge_reg 2e-4 \
  --reflectance_edge_uplift_reg 3e-3 \
  --reflectance_contrast_reg 2e-3 \
  --reflectance_highfreq_reg 3e-3 \
  --highlight_reflectance_reg 1e-3 \
  --residual_chroma_reg 5e-4 \
  --reflectance_detail_reg 1e-6 \
  --reflectance_decoder_reg 2e-5 \
  --reflectance_offset_lr 0.008 \
  --reflectance_decoder_lr 0.002 \
  --sg_smooth_reg 5e-5 \
  --b0_spatial_smooth_reg 0.0 \
  --residual_start_iter 3000 \
  --residual_ramp_iters 2500 \
  --residual_higherror_percentile 0.9 \
  --residual_highlight_percentile 0.92 \
  --enhancement_reflectance_reg 0.06 \
  --enhancement_degree_reg 0.2 \
  --enhancement_degree_global_reg 0.05 \
  --enhancement_smooth_reg 4e-4 \
  --enhancement_diff_start_iter 2350 \
  --enhancement_color_reg 0.06 \
  --enhancement_color_std_reg 0.02 \
  --enhancement_green_bias_reg 0.06 \
  --noise_residual_reg 1.0 \
  --artifact_residual_reg 0.35 \
  --noise_zero_mean_reg 0.03 \
  --noise_highfreq_reg 0.03 \
  --noise_dark_weight_reg 0.03 \
  --artifact_highlight_reg 0.0
```

Observed metrics (run date: 2026-05-18):

- Train lowlight: `PSNR=41.3628`, `SSIM=0.9324`, `LPIPS=0.3133`
- Train enhanced_gt: `PSNR=17.8513`, `SSIM=0.2632`, `LPIPS=0.6687`
- Test lowlight: `PSNR=40.4566`, `SSIM=0.9256`, `LPIPS=0.3211`
- Test enhanced_gt: `PSNR=17.8936`, `SSIM=0.2619`, `LPIPS=0.6757`


## ⚙️ Inference Arguments
| Argument            | Description                             |
| ------------------- | --------------------------------------- |
| `-m / --model_path` | Path to checkpoint                      |
| `--dataset_path`    | Dataset root                            |
| `--skip_train`      | Skip training split rendering           |
| `--skip_test`       | Skip test split rendering               |
| `--skip_optimize`   | Enable test-view rendering branch       |
| `--iteration`       | Load specific iteration (`-1` = latest) |
| `--infer_video`     | Export interpolated video               |




## 📂 Output Structure

```text
<model_path>/
  test/
    ours_<iter>/
      renders/
      render_reflectances/
      render_illuminations/
      render_enhanceds/
      render_depths/
      render_residuals/
      errors/
      gt/
      per_view_count.json
```


## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{sun2025ll,
  title={Ll-gaussian: Low-light scene reconstruction and enhancement via gaussian splatting for novel view synthesis},
  author={Sun, Hao and Yu, Fenggen and Xu, Huiyao and Zhang, Tao and Zou, Changqing},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={4261--4270},
  year={2025}
}
```
## 📜 LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## 🙏 Acknowledgement

We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work.
