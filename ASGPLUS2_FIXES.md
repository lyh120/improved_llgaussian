# asgplus2 实验经验回迁（2026-10-09）

修复当前 `asgplus2`，起点 `83d62ce`。保留本分支显式 R、ASG L、SG/context 增强结构、CIDNet/StableSR 先验选择、损失配置、训练预算和增密预算；没有把结构替换成此前官方对照实验中的 A。以下是工程修复，不是图像质量提升结论。

## 梯度权限

- R decoder 的 feature 和 offset 输入 detach，前向数值不变。
- 使用与主 R 图数值相同的辅助 R 图，冻结 rasterizer 的位置、相机变换、尺度、旋转、opacity 和屏幕坐标。
- R 的锐化、局部对比、平滑、颜色保护、detail/decoder/B0 正则及先验对 R 的小权重监督，通过参数白名单单独累积梯度，不加入主损失统一 backward。
- 显式 R 白名单：`base_log_reflectance`、`reflectance_offset_delta`、`mlp_reflectance_decoder`；MLP R 消融白名单为 `mlp_reflectance`。
- 增强分支的方向、共享特征、低光 L 和全部 rasterizer 几何输入 detach；增强颜色等监督使用 detach 的 R teacher。增强监督只更新 SG/context 增强参数。
- ASG 正则直接对全部显式 lobes 的幅度、两轴锐度求值，只更新 `illum_asg_amplitude`、`illum_asg_sharpness`。方向、切向量、底光仍由主监督学习。SG 模式正则只更新 `mlp_sg_illumination`。
- 每步从当前 optimizer 获取白名单参数，增点、剪枝替换参数后不会继续更新旧引用。
- 主低光重建保持 R/L 和几何可微；屏幕空间增密统计仍由主监督更新。

## 其他修复

| 问题 | 修复 |
|---|---|
| R 平滑权重再次乘 consistency 权重 | 独立使用 `reflectance_smooth_reg × L_Reflectance_Smooth` |
| 单 residual 正则使用 signed mean | 改为 `mean(abs(residual))`，防止正负抵消；前向和启用阶段沿用本分支配置 |
| 载入 pose 后丢失 requires_grad | 恢复可训练叶子，按 optim_pose 和 warmup 状态启用；缺 pose 时从训练相机初始化 |
| B0 初始化从测试图采样 | 点云/B0/sky 初始化改用训练相机 |
| test loss 是 step 前位姿，却保存 step 后候选 | 候选在 optimizer.step 前保存；只向 q/T backward |
| depth/residual 导出使用循环最后缓存 | train.py 和 render.py 统一使用选定位姿的 render_pkg_opt |
| ASG 零主轴不能构成正交框架 | 确定性回退到 +Z；近共线切向量使用正交回退 |
| loss 统计 fallback 无条件创建 CUDA tensor | 按已有统计 tensor 的设备创建零值，支持 CPU 检查 |

## “无效正则”的判断

模型目录下新增 `aux_gradient_audit.jsonl`，默认在阶段第一步和每 600 步记录原始损失、有效权重、加权值、梯度范数及 inactive 状态。可设置 `--aux_grad_audit_interval 100`。
`loss` 日志是主损失；`loss_total` 是主损失加辅助项的数值总和。

- 权重为零：关闭，不走辅助 backward。
- 有计算图但梯度为零：记录 `inactive=1`；可能是 hinge 已满足、饱和或无有效区域，不自动增加权重。
- 非零损失已 detach 或没有白名单路径：检查时报错，防止只记录数值而没有训练作用。
- 梯度非零但质量未改善：属于实验收益问题。此前 chair 的 sharp/color/stable 配置没有通过增强收益门槛，不能据此宣称更多约束会解决模糊或针状伪影。

保留本分支历史损失和权重以便兼容、消融，包括单边 highfreq；修复权限不代表证明其质量收益。本次没有自动叠加未获收益的实验约束，也没有放大权重掩盖问题。ASG 三项在审计样本上的加权梯度范数约 1e-8—1e-7，确实可微但很弱，应结合后续参数变化和质量结果判断。

## 已完成验证

1. 本地 CPU 单元测试 16 项通过：白名单、零权重、断路检测、增强 teacher、增密参数替换、ASG 公式及退化框架。
2. 远程 RTX 4090、真实 still2 相机、CUDA rasterizer：辅助 R 与主 R 最大差 0；逐项禁止参数梯度为零。
3. 主损失仍更新 offset、scale、共享特征、pose 和屏幕空间；辅助项不累积禁止参数梯度。独立 CUDA backward 的 atomic reduction 数值差已记录，不宣称逐 bit 相同。
4. 增强监督只更新增强分支；ASG 三项均产生非零梯度。
5. PLY + MLP 保存加载后的 R、L、增强 L 输出逐像素一致。
6. 原生 train.training 8 步短训练通过：单 residual、此前已核验的 StableSR cache、8 条梯度记录、iteration_8 保存。两次真实增点与剪枝每次各加/删 2 个 anchor。
7. 保存模型通过原生 render.py 三个 test 视角各 50 步位姿优化和图像导出。

本次没有重跑 2000+8000 步质量实验；旧报告指标不代表修复后结果。

## 可选 StableSR（2026-10-09）

默认保持 CIDNet；选择 StableSR：`python train.py ... --enhancement_prior stablesr`。
脚本支持 `bash scripts/train.sh ... --enhancement_prior stablesr`，批量入口支持
`bash scripts/single_train.sh stablesr`；改选 `cidnet` 即使用本分支 CIDNet 路径。
只在主训练选择 StableSR 且没有匹配缓存时才导入、加载扩散模型，CIDNet 路径不要求 StableSR 权重。

StableSR 的 get_SRModel、SD_refine 与 `E:/NTIRE/LL-Gaussian` 对齐，修正 VQGAN 的 submodules 配置路径。
保持 Turbo + VQGAN CFW、4 DDPM、seed 42、dec_w 0.75、adain、autocast、input 512、overlap 32、
VQGAN tile 1280/stride 1000、upscale 1。选择 StableSR 时验证这些配置，偏离时明确报错。
增益按发布代码 `int(0.45 / train_mean)`，不额外限定至少 1；输入先乘增益，
由原 SD_refine 在 bicubic resize 后截断，不在外部提前 clamp。不加曝光/gamma/其他色彩变换。

仅生成训练视角。采用 `diffusion_prior_llgaussian_<gain>_<fingerprint>` 独立缓存，
避免复用旧提前截断输入的缓存，也不覆盖旧实验先验。缓存 identity 包含训练输入、
生成器、gin/YAML 配置及两个权重的 SHA-256；加载前核验每张输出 PNG 哈希。
manifest 位于图像目录外，图像目录保持 `<uid>.png`，适配官方读取格式。
先验生成方式对齐发布版；使用先验的 R/L 配套梯度权限和监督配置仍属于 asgplus2。

StableSR 检查已通过：21 项 CPU 测试、两个 shell 入口语法检查、生成器/配置与本地 LL-Gaussian 对照。
远程用 still2 训练视角 DSC01613、gain 15 和已核验的两个官方权重实际采样，
恢复同一 RNG 状态后与冻结官方 SD_refine 的输出最大差为 0。
输入峰值 5.35294，确认没有提前 clamp；缓存 PNG 量化误差在 0.5/255 内，
缓存命中不重新加载扩散模型。结果见 `reports/asgplus2_fixes_20261009/stablesr_cuda_audit.json`。
两个脚本采用 LF，并通过 .gitattributes 保持 Linux 换行。

审计文件：`reports/asgplus2_fixes_20261009/cuda_audit.json`、`training_smoke_completed.json`、`aux_gradient_audit.jsonl`。
远程验证副本：`/home/liuyuhao/ll_further/asgplus2_fixes_20261009/code`。

复跑 CUDA 审计：

```bash
python scripts/audit_rl_gradients.py --eval --use_3D_filter \
  -s /path/to/still2 -m /path/to/asg_model --iteration 8000 \
  --output /path/to/cuda_audit.json
```

脚本用已有显式 ASG PLY 初始化、使用当前网络验证权限和保存加载，不转换已训练 SG 模型，不覆盖传入的模型目录。
