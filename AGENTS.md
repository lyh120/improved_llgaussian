# LL-Gaussian 项目开发指南

本文档为代码智能体提供项目开发规范。

## 项目概述

LL-Gaussian 是一个基于 Gaussian Splatting 的低光照场景重建与增强项目，使用 Python、PyTorch 与 CUDA 实现。

## 目录结构

```text
LL-Gaussian-ref-sgs/
├── train.py
├── render.py
├── arguments/
├── gaussian_renderer/
├── scene/
├── utils/
├── submodules/
└── scripts/
```

## 环境配置

```bash
conda create -n llgaussian python==3.10
conda activate llgaussian
pip install -r requirements.txt
pip install -e ./submodules/diff-gaussian-rasterization
pip install -e ./submodules/diff-gaussian-rasterization-fast
pip install -e ./submodules/diff-gaussian-rasterization-residual
pip install -e ./submodules/simple-knn
```

## 运行命令

```bash
bash scripts/train.sh -d ./dataset/LLRS-sRGB/{scene_name} -l {log_name} --gpu 0
python render.py -m ./backup/LLRS-sRGB/{scene_name}/{time} --dataset_path ./dataset/LLRS-sRGB/{scene_name} --skip_train
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-m/--model_path` | 模型检查点路径 |
| `--dataset_path` | 数据集根目录 |
| `--skip_train` | 跳过训练集渲染 |
| `--skip_test` | 跳过测试集渲染 |
| `--iteration` | 加载指定迭代，`-1` 表示最新 |
| `--infer_video` | 导出插值视频 |

## 代码风格

- 类名使用 PascalCase，函数名和变量名使用 snake_case。
- 导入顺序为标准库、第三方库、本地模块，组内按字母排序。
- 建议新增函数补充类型注解和简短的 Google 风格文档字符串。
- PyTorch 推理代码使用 `torch.no_grad()` 和 `model.eval()`。
- 尽量避免全局变量、循环字符串拼接和无意义注释。

## 本次修改记录

### 2026-04-30 SG 解耦第一阶段

- `arguments/__init__.py`：新增 `use_sg_illumination`、`sg_lobes`、`sg_lambda_min`、`sg_energy_reg`、`sg_smooth_reg`、`reflectance_consistency_reg` 等参数。
- `utils/sg_utils.py`：新增 SG 方向归一化、球面高斯求值、能量/锐度统计工具。
- `utils/loss_utils.py`：新增 SG energy、SG sharpness、reflectance consistency 损失入口。
- `scene/gaussian_model.py`：接入 `mlp_sg_illumination`、学习率调度、checkpoint 保存/加载与旧 optimizer 兼容恢复。
- `gaussian_renderer/__init__.py`：接入 SG illumination 求值，保持原有渲染输出字段不变，并返回 `sg_stats`。
- `train.py`：接入 SG 正则和日志项。
- `render.py`：输出 `sg_stats.json` 并在推理时打印当前 illumination 模式。

### 2026-04-30 B0 + SG 正式主路径

- `scene/gaussian_model.py`：新增每个 anchor 的 `B0` 参数 `base_log_reflectance`，主反射率改为 `R = exp(B0)`，不再让 `mlp_reflectance` 作为新训练默认主路径。
- `scene/gaussian_model.py`：`base_log_reflectance` 已接入点云初始化、anchor growing、anchor prune、PLY 保存/加载与训练 optimizer。
- `scene/gaussian_model.py`：新增 `illumination_mode` 与 `legacy_compatibility_mode`。新模型默认走 `sg`，旧 checkpoint 缺少 SG 或 B0 时进入 `legacy` 兼容模式。
- `gaussian_renderer/__init__.py`：主渲染路径默认使用 `B0 + SG`，只有兼容模式下才允许调用旧 `mlp_reflectance` 和 `mlp_illumination`。
- `train.py`：移除 warmup 阶段切回旧 illumination 的逻辑，继续沿用 SG 主路径，并补充 reflectance smoothness 约束。
- `render.py`：推理时明确打印当前是 `SG illumination and B0 reflectance` 还是 `legacy compatibility mode`。

### 训练测试启动说明（B0 + SG）

你现在可以直接发起新逻辑训练测试。以下命令按你的实际路径填写：

```bash
python train.py --eval -s /home/liuyuhao/ll_further/LL-Gaussian/dataset/LLRS-sRGB/chair -m /home/liuyuhao/ll_further/LL-Gaussian-sg/outputs/chair_sg_exp --gpu 0 --use_sg_illumination --illumination_mode sg --use_3D_filter
```

如果使用脚本批量训练，`scripts/train.sh` 已同步显式传入：

- `--use_sg_illumination`
- `--illumination_mode sg`

`scripts/single_train.sh` 调用 `scripts/train.sh`，因此会自动继承上述同步参数。

训练完成后（例如有 `/home/liuyuhao/ll_further/LL-Gaussian-sg/outputs/chair_sg_exp/point_cloud/iteration_30000`），可用如下命令测试渲染：

```bash
python render.py -m /home/liuyuhao/ll_further/LL-Gaussian-sg/outputs/chair_sg_exp --dataset_path /home/liuyuhao/ll_further/LL-Gaussian/dataset/LLRS-sRGB/chair --iteration 30000 --skip_train
```

## 修改记录总结

以下为本轮从“SG 解耦改造”到“B0 + SG 正式主路径”阶段，实际改动的项目文件统计：

1. `arguments/__init__.py`
2. `scene/gaussian_model.py`
3. `gaussian_renderer/__init__.py`
4. `train.py`
5. `render.py`
6. `utils/loss_utils.py`
7. `utils/sg_utils.py`（新增）
8. `scripts/train.sh`
9. `AGENTS.md`

10. `render.py`（新增 `--include_residual_render` 开关，默认最终渲染不带 residual）

## 最终渲染说明（Residual）

- 训练阶段：保留 `+ residual`，用于吸收瞬时噪声与难拟合误差。
- 最终渲染阶段：默认关闭 residual 参与输出，避免将噪声吸收分支带入最终结果。
- 如需导出 residual 渲染结果，可在 `render.py` 命令后追加：

```bash
--include_residual_render
```

## 修改记录总结

本轮围绕 SG 解耦、B0 反射率主路径、Residual 训练/渲染职责与可视化诊断，实际改动文件如下：

1. `arguments/__init__.py`
2. `scene/gaussian_model.py`
3. `gaussian_renderer/__init__.py`
4. `train.py`
5. `render.py`
6. `utils/loss_utils.py`
7. `utils/sg_utils.py`（新增）
8. `scripts/train.sh`
9. `AGENTS.md`

### 2026-04-30 Wandb Residual 可视化修正

- `train.py`：保留训练时 residual 原始数值参与 `I = R * L + residual` 的监督，不改损失与前向。
- `train.py`：将 wandb 中的 `residual_image` 改为诊断可视化图，使用 `abs(residual * enhance_ratio)` 后按当前帧最大值归一化，便于观察噪声吸收结构。
- `train.py`：新增 `residual_image_raw`，表示 residual 对最终结果的真实贡献量级，通常会较暗，这是正常现象。
- `train.py`：新增 `residual_abs_mean` 标量日志，便于判断 residual 分支是否完全塌缩。
- 说明：如果 wandb 中 `residual_image_raw` 接近纯黑，不代表 residual 无效；应优先结合 `residual_image` 与 `residual_abs_mean` 一起判断。

### 2026-04-30 Render 旧配置字段兼容修正

- `arguments/__init__.py`：新增旧字段到新字段的兼容回填逻辑，支持将 `num_sg -> sg_lobes`、`use_sg -> use_sg_illumination` 自动映射。
- `arguments/__init__.py`：当旧 `cfg_args` 缺少 `illumination_mode`、`sg_lambda_min`、`sg_energy_reg`、`sg_smooth_reg`、`reflectance_consistency_reg` 时，自动补默认值，避免渲染或恢复时报属性错误。
- `render.py`：在创建 `GaussianModel` 前使用 `getattr(...)` 做二次兜底，确保旧实验目录下的 `cfg_args` 也能完成渲染测试。
- 说明：如果 Linux 端仍报 `dataset.num_sg` 或类似字段错误，优先确认 `render.py` 与 `arguments/__init__.py` 已同步到最新版本。

### 2026-04-30 Linux 导入路径修正

- `render.py`：将项目根目录 `PROJECT_ROOT` 插入 `sys.path` 的动作前移到本地模块导入之前，避免 Linux 环境下 `from arguments import ModelParams` 命中错误模块或命名空间包。
- `train.py`：同步使用相同的 `PROJECT_ROOT` 路径初始化方式，保证训练与渲染入口的导入行为一致。
- 说明：如果仍出现 `ImportError: cannot import name 'ModelParams' from 'arguments'`，优先确认最新 `train.py`、`render.py`、`arguments/__init__.py` 已完整同步到 Linux 工作目录。

### 2026-04-30 Local Arguments 强制加载兜底

- `scene/__init__.py`：若普通 `from arguments import ModelParams` 失败，则直接从项目内 `arguments/__init__.py` 通过 `importlib.util.spec_from_file_location(...)` 加载本地模块。
- `render.py`：对 `ModelParams`、`PipelineParams`、`get_combined_args` 增加相同的本地文件加载兜底。
- `train.py`：对 `ModelParams`、`PipelineParams`、`OptimizationParams` 增加相同的本地文件加载兜底。
- 说明：该修复用于规避 Linux/conda 环境中同名第三方包、命名空间包或异常 `PYTHONPATH` 抢占 `arguments` 导入的问题。

### 2026-04-30 Test 渲染分支修正

- `render.py`：修正 `render_sets(...)` 中 test 渲染分支条件。
- 修正前：`--skip_train` 且默认不加 `--skip_optimize` 时，test 集实际上不会执行 `render_set(...)` 或 `render_set_optimize(...)`，表现为脚本加载完成后几乎不导出结果。
- 修正后：
  - `--skip_optimize` 为真时，直接执行普通 test 渲染 `render_set(...)`
  - `--skip_optimize` 为假时，执行带位姿优化的 test 渲染 `render_set_optimize(...)`
- 说明：现在 `python render.py ... --skip_train` 会按参数语义正常导出 test 结果。

### 2026-04-30 Interp 与 Depth 导出修正

- `render.py`：修正 depth 彩图导出时直接对 requires-grad tensor 调用 `.cpu().numpy()` 的问题，改为 `.detach().cpu().numpy()`，避免 test 渲染中断。
- `render.py`：修正 `interp` 渲染时 `gt` 未定义却仍尝试保存的错误，改为仅在非 `interp` 且 `gt` 存在时写入 GT 图像。
- `render.py`：修正插值视频生成时的目录名，从错误的 `render_enhanced` 改为实际输出目录 `render_enhanceds`。
- 说明：现在 `--infer_video` 路径能够更稳定地完成 test 渲染、interp 帧导出与视频合成。

### 2026-04-30 训练目标一致性修正

- `train.py`：将 SSIM 监督对象从仅 `reflectance * illumination` 改为完整重建 `image_tmp`，与 L1 主重建项保持一致。
- `train.py`：这样在 normal train 阶段，SSIM 也会正确监督 `reflectance * illumination + residual`，避免不同重建项看见的目标不一致。
- `train.py`：将 residual 正则从 `mean(residual_image)` 改为 `mean(abs(residual_image))`，避免正负 residual 互相抵消，保证正则真正约束 residual 幅度。
- 说明：该修正不会改变 warmup / normal train 的阶段划分，但会让主重建目标与 residual 抑制逻辑更自洽。

### 2026-04-30 Depth Prior 类型兼容修正

- `scene/__init__.py`：修正 `depth_piror_generator(...)` 对 `DepthAnythingV2.infer_image(...)` 返回值的假设，兼容 `numpy.ndarray` 与 `torch.Tensor` 两种类型。
- `scene/__init__.py`：当返回 numpy 时自动转换为 `float32 torch.Tensor`，并在二维深度图场景下补 `unsqueeze(0)` 变为 `[1, H, W]`，保持后续 `minmax_normalize(...)` 与 loss 输入形状一致。
- `scene/__init__.py`：将 depth prior 移动到训练设备（CUDA 可用时）以避免后续与渲染深度张量的设备不一致问题。
- 说明：该修复直接解决报错 `AttributeError: 'numpy.ndarray' object has no attribute 'unsqueeze'`，并提升不同 DepthAnything 版本下的兼容性。

### 2026-04-30 Interp 视频写出修正（FPS 参数报错）

- `render.py`：将 `images_to_video(...)` 从 `imageio.mimwrite(..., fps=...)` 改为 OpenCV `VideoWriter`（`mp4v`）写 MP4，避免环境中 `imageio` 错选 TIFF 插件导致 `TypeError: TiffWriter.write() got an unexpected keyword argument 'fps'`。
- `render.py`：视频写出时增加空帧目录检查、灰度帧转三通道、分辨率不一致时自动 resize、RGB->BGR 转换，保证插值视频导出稳定。
- 说明：该修复不会改变渲染帧内容，只修复最后的视频封装环节。

### 2026-05-01 Render Test 输出与 GT 指标补充

- `render.py`：明确 `test/ours_xxx/renders` 为最终常规渲染输出，`test/ours_xxx/render_enhanceds` 为最终增强渲染输出；`renders(enhanced)` 仅是 `render * 30` 的可视化亮图，不作为正式增强结果指标。
- `render.py`：在 `--skip_optimize` 的 test 渲染路径下，新增两组自动评估：
  - `metrics_lowlight.json`：`renders` 对测试视角原始低光图 `gt`
  - `metrics_enhanced_gt.json`：`render_enhanceds` 对数据集目录 `source_path/gt/images`
- `render.py`：评估指标包含 `PSNR / SSIM / LPIPS`，同时输出 summary 和 per-view 结果。
- 说明：`interp` 路径不做 GT 指标计算，因为插值相机不对应真实采样视图。

### 2026-05-01 Train 指标评估开关

- `render.py`：新增命令行参数 `--eval_train_metrics`，用于在渲染 train 集时同时计算并保存指标 JSON。
- `render.py`：开启后 train 目录同样输出 `metrics_lowlight.json` 和 `metrics_enhanced_gt.json`，并在终端打印 summary（平均 `PSNR / SSIM / LPIPS`）。
- 说明：默认不加该参数时，train 仍只导图不算指标，以节省渲染时间。

### 2026-05-03 Reflectance 解耦纠偏（最小修复版）

- `train.py`：恢复主重建中 `reflectance_image * illumination_image (+ residual)` 对 reflectance 的梯度，不再在 `image_tmp` 中对 `reflectance_image` 做 `detach()`。
- `train.py`：恢复 `L_diff` 第二项对 reflectance 的监督，只保留第一项中对 reflectance 的 `detach()`，避免增强支路完全切断 R 的有效学习信号。
- `train.py`：移除 `L_reflectance_lowfreq` 低频重建补项，避免通过低频 reflectance 重建 GT 的方式把 illumination 进一步推糊。
- `train.py`：将 residual 正则调度下限从 `0.5` 提高到 `1.0`，减轻中后期 residual 过度吸收 reflectance 噪点的问题。
- `arguments/__init__.py`：将默认 `reflectance_consistency_reg`、`reflectance_smooth_reg` 从 `5e-3` 回调到更稳的 `1e-3`。
- `arguments/__init__.py`：将默认 `b0_spatial_smooth_reg` 设为 `0.0`，暂时关闭当前随机 pair 近似版 B0 空间平滑，避免不稳定副作用。

### 2026-05-03 反射率解耦质量改进（三阶段）

#### 改进 1：强化 R 正则化

- `arguments/__init__.py`：新增 `reflectance_smooth_reg=5e-3`（参数化，原硬编码 `5e-4`）、`b0_lr=0.001`（原继承 `feature_lr=0.0075`）、`b0_spatial_smooth_reg=1e-3`；`reflectance_consistency_reg` 从 `1e-4` 提升至 `5e-3`；`_backfill_model_compatibility` 同步补新参数默认值。
- `scene/gaussian_model.py`：所有 `training_setup` 分支的 `_base_log_reflectance` 学习率改为 `training_args.b0_lr`；新增 `b0_scheduler_args` 调度器；`update_learning_rate` 中加入 B0 LR 更新。
- `utils/loss_utils.py`：新增 `L_B0_Spatial_Smooth` 函数，基于 anchor KNN 的 B0 空间平滑损失。
- `train.py`：`L_reflectance_smooth` 权重从硬编码 `5e-4` 改为 `dataset.reflectance_smooth_reg`；新增 `L_b0_spatial_smooth` 损失项。

#### 改进 2：梯度流隔离

- `train.py`：非 warmup 分支主重建 `image_tmp` 中 `reflectance_image` 改为 `reflectance_image.detach()`，阻止 L1/SSIM 梯度回传到 R。
- `train.py`：`L_diff` 第二项 `reflectance_image` 改为 `reflectance_image.detach()`，阻止 StableSR 伪 GT 噪声梯度污染 B0。
- `train.py`：新增 `L_reflectance_lowfreq` 低频反射率监督（对 reflectance 做均值池化后 L1 约束），引导 R 只学低频结构。

#### 改进 3：B0 初始化改进

- `scene/gaussian_model.py`：新增 `_estimate_initial_b0` 方法，基于第一帧训练图像做 Retinex 初始化：将 anchor 投影到相机像素，取 `R = pixel / max_channel(pixel)`，`B0 = log(R)`，替代全零初始化。
- `scene/gaussian_model.py`：`create_from_pcd` 中 `base_log_reflectance` 初始化从 `torch.zeros(...)` 改为调用 `_estimate_initial_b0(fused_point_cloud, cameras)`。
- `scene/gaussian_model.py`：`anchor_growing` 中新 anchor 的 B0 从 `scatter_max` 改为 `scatter_mean`，取邻域均值而非最大值，避免新 anchor 的 B0 偏高导致 R 过亮。
- `scene/gaussian_model.py`：导入新增 `scatter_mean`。

### 2026-05-03 ȥ�������� Residual ����������

- `arguments/__init__.py`����Ĭ�� `reflectance_consistency_reg` �� `1e-3` �µ��� `5e-4`��`reflectance_smooth_reg` �� `1e-3` �µ��� `3e-4`�����Ͷ� `R` �Ĺ�ǿƽ����һ����ѹ�ơ�
- `arguments/__init__.py`����Ĭ�� `sg_smooth_reg` �� `1e-4` �µ��� `5e-5`������ SG illumination ���ȵ�Ƶ����
- `arguments/__init__.py`������ `residual_start_iter`��Ĭ�ϱ�׼ 30k ѵ���� `12000` ֮������� residual �����������ؽ����̳� 8k ѵ��������������ʽ�ĳ� `3500`��
- `train.py`���� `L_smooth` Ȩ�ش� `1e-3` �µ��� `5e-4`������ illumination ������
- `train.py`���� enhancement ��ص� `L_degree` �� `L_smooth_enhancement` Ȩ���µ���������ǿ�ල������ `L` ѹ�ɹ�ƽ���ṹ��
- `train.py`������ `residual_active` �׶ο��ƣ��� `iteration < residual_start_iter` ʱ������ residual ��֧��ǰ�����ɣ�Ҳ������� `image_tmp = R * L + residual` �����ؽ���
- `train.py`������ `residual_active` Ϊ��ʱ������ `L_residual_reg` �� residual scaling ���򣬱��� residual ����ӹܸ�Ƶ��
- `train.py`��wandb ���� `residual_enabled` ���������ڼ�� residual �Ƿ��Ѿ���������ѵ����
- `train.py`������ `_visible_count_value` �� `_visible_count_per_view`������ѵ�������������׶� `visible_count` �����ǵ�������ʱ���µ� `TypeError: 'int' object is not iterable`��
- `train.py`��`render_sets(...)` ������ test pose optimize ·������ `0` ���� train ��Ⱦ�õ��� `visible_count`����֤ѵ��������������ȶ���

### 2026-05-03 Reflectance ȥ�������������ڶ��֣�

- `arguments/__init__.py`����Ĭ�� `reflectance_consistency_reg` ��һ���µ��� `2e-4`��`reflectance_smooth_reg` ��һ���µ��� `1e-4`����ȷ�ѱ������ȼ����ڡ�R �������������Ǽ�������ѹ�롣
- `scene/gaussian_model.py`������ `_estimate_initial_b0()`������ֱ��ʹ�õ�һ֡ `image / max_channel(image)` ��ԭʼͶӰֵ������һ�ξֲ���ֵƽ�������� `reflectance_map + 0.25 * (reflectance_map - blur)` ������������ǿ������ B0 ��ʼ����Ȼ��Ƶ�������⡣
- `train.py`���� `L_diff` �еڶ��`illumination_enhanced.detach() * reflectance`����Ȩ�ش� `0.2` �µ��� `0.05`������ StableSR α GT �� reflectance ��ƽ��ǣ����
- `train.py`������ `residual_abs_mean_raw` �� `residual_abs_mean_used` ��������������� residual ǰ��ԭʼ�����������������ؽ���ʧ�� residual �������������� residual �Ƿ��Ѿ�����ѵ����
- `train.py`��debug ���ͬ����Ϊ��ӡ `residual_abs_mean_raw` �� `residual_abs_mean_used`�������� `residual_image_raw_mean` ���ں���Աȡ�
- `scene/gaussian_model.py`������ `import torch.nn.functional as F`���޸� `_estimate_initial_b0()` ��ʹ�� `F.avg_pool2d(...)` ʱ�� `NameError: name 'F' is not defined`�����ǵڶ���ȥ����ʼ���Ķ���ı�Ҫ���롣

### 2026-05-03 �Ҷ� L ǰ���µ������赼����

- `arguments/__init__.py`����Ĭ�� `reflectance_consistency_reg` �� `reflectance_smooth_reg` ��һ���µ��� `5e-5`���������� R �Ĺ��ȵ�Ƶ����
- `arguments/__init__.py`������ `highlight_reflectance_reg = 1e-4`���������ơ����Ҳʡ��� reflectance ����������
- `utils/loss_utils.py`������ `L_Reflectance_Highlight(...)`��ͨ��������ֵ��ɫ��ƫ������ϳͷ� reflectance �еĸ�����ɫ�쳣�
- `scene/gaussian_model.py`���� `_estimate_initial_b0()` �� detail ע��ϵ���� `0.25` ��ߵ� `0.4`��������ǿ��֡ B0 ��ʼ���ľֲ��Աȱ���������
- `train.py`������ `L_Reflectance_Highlight`���� warmup �� normal train �ж��� `dataset.highlight_reflectance_reg` ��Ȩ�ؼ��� loss��
- `train.py`���� `L_diff` �ڶ��`illumination_enhanced.detach() * reflectance`��Ȩ�ش� `0.05` �����µ��� `0.02`����һ������ StableSR α GT �� reflectance ��ƽ��ǣ����
- `train.py`������ `reflectance_highlight_mean` �� `residual_chroma_mean` ��־�����ڹ۲��ɫ�����Ƿ�� R �� residual ת�ơ�
- `train.py`�������Ҷ� illumination ���ṹ���䣬���� RGB illumination ���졣

### 2026-05-03 ���ߴ� R �� Residual �赼����ǿ�棩

- `arguments/__init__.py`����Ĭ�� `highlight_reflectance_reg` �� `1e-4` ��ߵ� `5e-4`����ǿ�� reflectance �С����Ҳʡ��쳣���ѹ�����ȡ�
- `arguments/__init__.py`������ `residual_chroma_reg = 5e-5`�����ڸ� residual һ���ǳ���������ʶȲ���������
- `utils/loss_utils.py`������ `L_Residual_Chroma_Boost(...)`���� reflectance ������ residual �Ĳʶȸ�����������������ɫ���ߴ� R �� residual Ǩ�ơ�
- `train.py`���� residual ������ʱ���� `L_Residual_Chroma_Boost`���� `dataset.residual_chroma_reg` ��Ȩ�ؼ����� loss��
- `train.py`������ `residual_chroma_boost` ��־�� debug ������������� residual ��ǰ���ڱ������������Ѿ���ʼ�����е������ʶȲ�����

### 2026-05-03 Reflectance ��������ǿ������Ǩ���ټ�ǿ

- `arguments/__init__.py`����Ĭ�� `reflectance_smooth_reg` ��Ϊ `0.0`����һ����� R ��ͼ��ƽ���������� `reflectance_consistency_reg` �µ��� `2e-5`�������������ȶ����á�
- `arguments/__init__.py`������ `reflectance_edge_reg = 2e-4`�����ڹ��� R ��������ͼ���еĽṹ��Ե��
- `arguments/__init__.py`����Ĭ�� `highlight_reflectance_reg` ��ߵ� `1e-3`���� `residual_chroma_reg` ��ߵ� `5e-4`����ǿ��ɫ���ߴ� R �� residual Ǩ�Ƶ���������
- `utils/loss_utils.py`������ `L_Reflectance_Edge(...)`���� reflectance �Ҷ��ݶ���͹�ͼ�Ҷ��ݶȵĹ�һ�����������ṹ�ල������ֻ���� smooth ��Ȼ�޷���������
- `train.py`������ `L_Reflectance_Edge`���� warmup �� normal train �ж��� `dataset.reflectance_edge_reg` ������ loss��
- `train.py`������ `reflectance_edge_mean` �� wandb �� debug ��������ڹ۲� R �Ľṹ��Ե�ල�Ƿ���Ч��

### 2026-05-03 Reflectance 表达粒度升级（anchor 底色 + offset 细节）

- `scene/gaussian_model.py`：在保留 `R = exp(B0)` 主路径不变的前提下，新增 `_reflectance_offset_delta`，将 reflectance 表达升级为“anchor 级底色 + offset 级 log-detail”，用于提升局部纹理和边缘清晰度。
- `scene/gaussian_model.py`：`get_reflectance_with_detail` 使用 `exp(B0 + 0.35 * tanh(detail))` 构造最终 reflectance，既增加局部表达能力，又限制细节分支过度吸收彩色亮斑。
- `scene/gaussian_model.py`：`capture/restore`、`create_from_pcd`、optimizer 参数组、学习率调度、anchor growing、anchor prune、PLY 保存/加载 均已接入 `_reflectance_offset_delta`，旧 checkpoint 和旧 PLY 缺少该字段时自动以全零 detail 兼容恢复。
- `gaussian_renderer/__init__.py`：主渲染路径和 fast 路径不再简单重复 anchor 级 `R`，而是改为使用 `get_reflectance_with_detail` 的 offset 级 reflectance。
- `arguments/__init__.py`：新增 `reflectance_offset_lr=0.002`，用于单独控制 reflectance detail 的学习速度；新增 `reflectance_detail_reg=1e-5`，用于约束 detail 分支不过度放大。
- `train.py`：新增 `L_reflectance_detail`，对 `tanh(_reflectance_offset_delta)` 的幅度做轻量正则，既允许 R 变清晰，又限制 detail 重新学回彩色亮斑。
- `train.py`：新增 `reflectance_detail_mean` 日志，便于观察 offset 级 reflectance detail 是否真正开始承担清晰度而不是继续由 residual 或 illumination 代偿。

### 2026-05-03 Reflectance detail PLY 兼容修正

- `scene/gaussian_model.py`：修正 `load_ply_sparse_gaussian()` 读取 `b0_*` 字段时的匹配条件，`b0_detail_*` 不再被误并入 `base_log_reflectance`，避免加载新 PLY 后 `B0` 通道数错误膨胀。
- `scene/gaussian_model.py`：`get_reflectance_with_detail` 增加防御式兼容处理；若历史中间产物导致 `base_log_reflectance` 通道数异常大于 3，则仅取前 3 通道参与 `exp(B0 + detail)`，避免训练后自动渲染阶段直接 shape mismatch 崩溃。

### 2026-05-03 fix8 Reflectance 高清化改进

- `arguments/__init__.py`：将默认 `reflectance_offset_lr` 从 `0.002` 提高到 `0.006`，将 `reflectance_detail_reg` 从 `1e-5` 降到 `2e-6`，让 offset 级 reflectance detail 更容易学到局部纹理。
- `arguments/__init__.py`：新增 `reflectance_edge_uplift_reg = 1e-3`，并在 `_backfill_model_compatibility` 中补默认值，旧配置恢复时也能安全读取。
- `scene/gaussian_model.py`：将 `reflectance_detail_scale` 从 `0.35` 提高到 `0.8`，保持 `exp(B0 + scale * tanh(detail))` 的有界 detail 结构不变。
- `utils/loss_utils.py`：新增 `L_Reflectance_Edge_Uplift(...)`，只惩罚 reflectance 边缘弱于低光输入结构的位置，用 one-sided hinge 鼓励 R 补边，不压制已经更清晰的区域。
- `train.py`：接入 `L_Reflectance_Edge_Uplift`，以 `dataset.reflectance_edge_uplift_reg` 加入 warmup 和 normal train 的总损失。
- `train.py`：将 `L_diff` 第二项系数从 `0.02` 改为 `0.0`，默认不再让 StableSR 伪 GT 直接回传梯度到 reflectance，只保留增强光照分支监督。
- `train.py`：新增 `reflectance_edge_uplift_mean` 到 wandb 与 debug 输出，便于观察 R 的边缘补强是否真正生效。
- 训练建议：8k 诊断训练中将 `residual_start_iter` 推迟到 `6000`，让 `R * L` 先承担主要结构重建，再允许 residual 做少量补偿。

### 2026-05-03 fix8.1 Reflectance 补边增强与 Residual 线性爬坡

- `arguments/__init__.py`：将 `reflectance_edge_uplift_reg` 从 `1e-3` 提高到 `3e-3`，将 `reflectance_detail_reg` 从 `2e-6` 继续降到 `1e-6`，并将 `reflectance_offset_lr` 从 `0.006` 提高到 `0.008`，进一步放大 offset-detail 对 R 清晰度的贡献。
- `arguments/__init__.py`：新增 `residual_ramp_iters = 2000`，用于控制 residual 从启用到全量参与主重建的线性爬坡时长。
- `scene/gaussian_model.py`：将 `reflectance_detail_scale` 从 `0.8` 提高到 `1.1`，增强 `exp(B0 + scale * tanh(detail))` 中 detail 对 reflectance 局部对比和纹理的影响。
- `utils/loss_utils.py`：重写 `L_Reflectance_Edge_Uplift(...)`，改为基于归一化梯度的 one-sided hinge；只惩罚 R 边缘弱于 GT 结构边缘的区域，避免因为原始梯度量级过小导致补边损失长期几乎为零。
- `train.py`：将 residual 从“到点硬开”改为“更早启用、线性爬坡”，通过 `residual_mix_weight` 控制 `render_residual` 进入 `image_tmp` 的比例，减轻一启用就出现的大幅 spike。
- `train.py`：新增 `residual_mix_weight` 到 wandb 与 debug 输出，便于检查 residual 介入时机和爬坡状态是否合理。

### 2026-05-03 fix10 反射率优先锐化

- `scene/gaussian_model.py`：新增 view-independent `mlp_reflectance_decoder`，输入仅使用 anchor 特征与 offset 局部几何，不接 view direction、不接 illumination；最终 reflectance 改为 `exp(B0 + detail) * (1 + 0.25 * tanh(decoder_out))` 后再 clamp。
- `scene/gaussian_model.py`：将 reflectance decoder 接入 `eval/train`、optimizer 参数组、学习率调度、freeze 逻辑，以及 split/unite checkpoint 的保存与恢复链路。
- `gaussian_renderer/__init__.py`：主渲染路径与 fast 路径统一改为使用 decoder-refined reflectance，不再仅依赖 `B0 + offset-detail` 的参数表表达。
- `utils/loss_utils.py`：新增 `L_Reflectance_LocalContrast(...)`，使用 GT 灰度局部对比度对 reflectance 做单边补强；新增 `build_residual_hard_mask(...)`，从高重建误差区域与高彩度亮区构造 residual 的局部补偿 mask。
- `train.py`：接入 `reflectance_contrast_reg` 与 `reflectance_decoder_reg` 两个新损失；新增 `reflectance_contrast_mean`、`reflectance_decoder_mean`、`residual_hardmask_coverage` 日志。
- `train.py`：enhancement 的 `L_diff` 第一项改为阶段性衰减权重，继续保持第二项为 `0.0`，降低 StableSR 伪 GT 对 reflectance 的长期平滑牵引。
- `train.py`：residual 混入改为 `residual * residual_mix_weight * hard_mask`，默认更早启用（`residual_start_iter=3000`、`residual_ramp_iters=2500`），但只在局部困难区域吸收误差，避免重新参与整图低频拟合。
- `arguments/__init__.py`：新增 `reflectance_decoder_lr=0.004`、`reflectance_decoder_reg=1e-5`、`reflectance_contrast_reg=2e-3`、`residual_hardmask_percentile=0.8`，并同步更新兼容回填默认值。
