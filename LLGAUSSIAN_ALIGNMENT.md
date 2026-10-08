# R/L 表达 + LL-Gaussian 原版监督

2026-10-08 当前训练默认 `--supervision_profile llgaussian`。这份说明取代此前实验路径作为新训练入口；旧配置缺少此字段时自动恢复 `custom`，保持历史 checkpoint 的前向语义。

保留的表达设计：R 使用 anchor B0、offset log-detail、无视角 R decoder；L 使用 SG（默认）或显式选择的 ASG。保留训练视角的 B0 初始化、SG 参数合法性变换、PLY/MLP 保存加载和旧字段兼容。没有额外 SG、B0 或 R 图像损失，参数合法性由表达中的归一化、有界变换和 clamp 保证。

原版监督位于 `utils/llgaussian_objective.py`，数值和梯度通过 `tests/fixtures/llgaussian_loss_excerpt.txt`（直接摘自 `E:\NTIRE\LL-Gaussian\train.py`）核对：

- 主重建：先分别 splat R、L，再乘图像，训练时加原版单 residual；weighted L1 权重 0.7，intrinsic DSSIM 权重 0.3。
- 光照：原版 L_Illu；第 1000 步启用 L_Smooth，kernel 9、权重 0.001。原版 depth Pearson patch loss 同时启用，patch 128、采样比例 0.5、权重 0.15。
- 原版 Gaussian volume 0.01；单 residual 从主训练开始参与，无 hard mask、延迟或 ramp，均值抑制从 2 线性降至 0.5，residual volume 0.05。原版 residual 输出 sigmoid 非负，因此均值等于 L1 范数。
- 增强模块恢复原版一层隐藏层 MLP，输入 detach 的 anchor feature 与 SG 输出 illumination feature；第 1000 步启用 degree 两项 0.2/0.05 和 smooth 0.0005；第 2000 步启用 StableSR prior，L 项权重 1、R 项 0.2，detach 方向与原版一致，无退火。
- R 锐化、局部对比、高频、highlight、跨视角一致性、B0 空间平滑、detail/decoder 正则、颜色均值/方差/green bias、gain/edge enhancement、额外 footprint 上界、粗分辨率几何监督均不在此路径执行。
- 原版监督使用原版 rasterizer 梯度权限，增强 illumination raster 只 detach means3D；其 opacity/covariance 梯度保留。上一轮 R 锐化损失在此路径不存在；旧 `custom` 路径保留其仅 R 权限隔离。
- Densification 恢复原版随机候选筛选，无 Top-K/global growth 上限，无 never-visible prune；保留新增 R 参数的继承与 checkpoint 兼容。

chair 复现：`bash scripts/validate_llrs_chair_sg.sh DATASET MODEL_PATH GPU`。按原版 `single_train.sh` 的配置：不单独 warmup，8000 步，voxel 0.001、prune 0.1、feat 32、residual embedding 32、update_from 1000/update_until 5000、颜色 LR 0.04→0.00025，其他 scheduler 8000。原版另有可选 2000 步 warmup；没有把它追加到论文 8000 步预算。

StableSR helper 与原版相同（VQGAN 配置路径用等价 symlink 兼容），`configs/config.gin` 与参考版本相同，冻结 StableSR turbo + VQGAN，训练图乘场景增益后生成 32 张 prior。已有本任务 prior 仅在每张 prior 与训练输入 SHA256 一致时复用；不使用正常光 GT 训练，不使用 CIDNet。Depth Anything V2 ViT-L 与原版相同。

test 采用 `render.py` 默认 50 步位姿优化，固定 pose 与训练入口 500 步结果单列。正常光增强按论文 LAB L 仿射对齐评估，保留 raw 结果；真实低光三指标独立报告。全部 5 个 held-out test 视角、原始 1107×798 分辨率、不裁剪。LPIPS VGG 输入 `[0,1]`、`normalize=False`，跟随参考发布代码。

原版 `render_fast` 是 splat(R×enhanced L)，与论文/主渲染的 splat(R)×splat(enhanced L) 一般不相等。正式指标与图片始终来自主 renderer。RTX4090 与论文 V100 不同；R/L 网络也不同，不能宣称参数量、FLOPs、硬件时间或整模型与论文完全相同。指标与逐图视觉结论以本轮完整报告为准。
