# LL-Gaussian v2 开发约束

## 当前架构

- 模型格式固定为 `MODEL_FORMAT_VERSION=2`，旧 checkpoint、PLY 和旧 `cfg_args` 必须明确拒绝。
- Scaffold-GS 只允许 `mlp_opacity`、`mlp_cov` 和可选 `mlp_feature_bank`。
- Appearance 必须由 `scene/explicit_appearance.py` 中的显式 tensor 表达：层次化 R、单叶灰度 ASG、独立单叶 RGB SG。
- 增强分支使用有界 RGB diffuse 与独立单叶加性 RGB SG：`B=R.detach()*D`、`C_enh=B+(1-B)*S(v)`；禁止恢复 `enhancement_prior/R` 无界增益。
- 禁止重新加入 color/reflectance/illumination/enhancement decoder、appearance embedding、controller、residual、noise 或 artifact 网络。
- 禁止加入 `mlp/sg/asg/legacy` 等运行时 appearance 模式。

## 梯度边界

- low photo：更新 R、主 ASG 与 Scaffold 几何。
- reflectance reconstruction：使用诊断 R 与 stop-gradient 主 L，只更新 R base/detail。
- illumination photo 与 edge-aware TV：只更新主 ASG。
- enhanced photo 与 enhanced-illumination photo：使用有界 diffuse/additive-SG 组合，并 detach R 与全部 Scaffold 几何，只更新增强 diffuse/SG。
- depth 与 scale：只更新 Scaffold 几何。
- R/L/L-enhanced 诊断 rasterization 必须 detach 几何与 coverage，并除以 coverage 得到 covered mean；禁止把 raw alpha-premultiplied accumulation 用作监督或可视化。
- `render_depth` 必须是由 live geometry 计算的 expected depth `sum(w*z)/sum(w)`；禁止 detach depth 的分子或 coverage 分母。

## Loss 与参数

只允许六个具名 loss 参数：

```text
lambda_dssim=0.2
lambda_scaling=0.01
lambda_reflectance_reconstruction=1.0
lambda_illumination=1.0
lambda_enhanced=1.0
lambda_depth=1.0
```

所有显式 appearance tensor 共用：

```text
explicit_appearance_lr_init=0.008
explicit_appearance_lr_final=0.00005
```

除 `NUMERICAL_EPS` 外，不得在 appearance 或 loss 中加入隐藏倍率。Scaffold/3DGS 原生几何常数必须提升为带来源说明的命名常量，不能与 appearance/loss 参数混用。

## Scaffold 初始化与时序

- `prune_ratio=1.0` 是 v2 默认值，表示保留完整输入点云；它不是训练期 anchor pruning 参数。
- 启用 anchor growth 时必须满足 `update_until < iterations`，为继承得到的 appearance 留出无增密精修阶段。
- `max_new_anchors_per_update=0` 明确关闭 growth；`max_pruned_anchors_per_update=0` 明确关闭训练期 pruning。
- 启用 pruning 时应满足 `prune_from_iter < min(update_until, iterations)`，否则不会产生有效 pruning 事件。
- 多视图 appearance 初始化必须做逐像素前景深度筛选；同一像素后的隐藏 anchor-offset 不得作为 R/L 观测。
- warmup 到 main 只清空 interval-local 的 `offset_gradient_accum/offset_denom`，不得清空 opacity、visibility、birth iteration 或 optimizer state。
- checkpoint 必须保存并校验 warmup enable/boundary/transition 状态；恢复命令不得静默改变阶段语义。

## Prior

- 默认缓存位于场景内：CIDNet 使用 `cidnet_prior/`，StableSR 使用
  `diffusion_prior_<显式输入增益>/`，DepthAnything 使用 `depth_maps/`。
- `train.py` 启动时复用已有 v2 manifest；目录缺失或为空时分别调用预处理脚本生成一次。
- 非空但缺少 v2 manifest 的目录必须报错，不得覆盖；训练迭代期间不得刷新或重复推理。
- `--enhancement_prior_backend` 可选择 `cidnet` 或 `stablesr`；两者只生成统一
  `enhancement_rgb` 固定 prior，不进入训练图。StableSR 输入亮度使用显式
  `--stablesr_input_gain`，禁止恢复 `0.45/mean` 一类隐藏倍率。
- CIDNet 位于 `submodules/HVI-CIDNet`，StableSR 位于 `submodules/StableSR`，
  DepthAnything 位于 `submodules/Depth-Anything-V2`。
- manifest 以 `image_name` 为唯一键；缺失、重复、额外、尺寸错误、dtype 错误或非有限值必须立即失败。

## 修改与验收

- Python 使用 snake_case，类使用 PascalCase；新增公共函数写简短 docstring。
- 编辑文件使用 `apply_patch`，保留用户无关改动。
- 首选 `rg` 搜索。
- 修改 appearance、renderer、loss、grow/prune 或格式时，必须同步增补测试；coverage 去预乘、expected-depth 梯度和非法增密时序必须有回归覆盖。
- 提交前运行：

```bash
python -m unittest discover -s tests -v
python -m py_compile train.py render.py arguments/__init__.py gaussian_renderer/__init__.py scene/__init__.py scene/explicit_appearance.py scene/gaussian_model.py scene/gaussian_model_v2.py utils/loss_utils.py utils/prior_utils.py
```

- CUDA 环境还需运行 rasterizer 梯度隔离、checkpoint 恢复、短训练与 train/render smoke test。
- 质量下降时优先检查初始化、可见性、Gaussian 空间组合和梯度隔离；不得用 decoder、residual、隐藏倍率或 loss 堆叠补救。
