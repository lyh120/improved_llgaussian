处理问题 2-7：可直接修的修掉，研究性问题整理成消融实验

## 前置：修正我上轮的一个错误结论
复查 `utils/sg_utils.py:119-131` 发现：`evaluate_anisotropic_spherical_gaussians` 返回的 stats 里 `sg_energy`/`sg_lambda_mean` 是 `asg_energy`/`asg_lambda_mean` 的**同值别名**（同为 amplitudes.mean()/bandwidth.mean()），并非我上轮所说的"恒为 0"。因此上轮 wandb 清理中"ASG 模式下 sg_* 死监控"的**理由错了，但处理结果仍然正确**（它们是完全重复的键，门控掉正是去重）。需要：AGENTS.md 中上轮记录的错误理由改为"重复别名"；本轮回复中已向你说明。

## 一、代码修复

### 1.（问题2）isolate=False 时跳过 sharpen 冗余拼接
`gaussian_renderer/__init__.py:267-272`：删除 `reflectance_sharpen = reflectance.detach()` 兜底，改为两个分支——isolate 时按现结构 cat/split（含 sharpen 3 通道），非 isolate 时不含 sharpen 通道（split 尺寸 [6,3,3,1,3,7,3]），`reflectance_sharpen` 保持 None 返回。下游 render() 仅在 isolate 时消费该值，train.py/render.py 已按标志处理，无行为变化。

### 2.（问题2/3）新增 isolate 手动开关，支持消融
- `arguments/__init__.py` ModelParams + backfill 表加 `disable_reflectance_grad_isolation = False`（ParamGroup 自动生成 `--disable_reflectance_grad_isolation` store_true 参数）。
- `train.py:740` 判定改为 `(not dataset.disable_reflectance_grad_isolation) and any(...)`。默认行为不变；置位后四个正则走旧的非隔离路径（A7 消融臂需要）。

### 3.（问题5）ASG 多 lobe 合并 mean→sum
`utils/sg_utils.py:109`：`lobes.mean(dim=2)` → `lobes.sum(dim=2)` + 注释。默认 `asg_lobes=1` 下数学等价（单元素 mean==sum），现有训练零影响；将来 lobes>1 时各 lobe 按自身幅度叠加而不是被 1/lobes 稀释。**SG 路径（`evaluate_spherical_gaussians`，sg_lobes 默认 4）不动**——mean 是已训练的既有语义，改动会破坏旧 SG run 可复现性。新增 `tests/test_sg_utils.py`：两个相同 lobe 的响应 ≈ 单 lobe 的 2 倍（bias 取极小值避开 clamp）。

### 4.（问题6）arguments 双默认值表加同步守卫
新增 `tests/test_arguments_backfill.py`：调用 `_backfill_model_compatibility({})` 拿到完整默认值表，与 `ModelParams(parser)`/`OptimizationParams(parser)` 实例属性逐一比对（含缺失检测），任何一边改默认值漏同步立即测试失败。不做侵入式重构，用测试锁死同步。

### 5.（问题6）README 标注历史配置已过时
在 README "Reproducible Best Setting" 与 "CIDNet Single Dense" 两节前各加一行说明：以下命令为 SG 时代记录，当前代码默认 `illumination_mode=asg + reflectance_mode=explicit`，照跑会走不同路径；当前 ASG 推荐配置见 `scripts/run_ablation_9.sh` 的 COMMON_ARGS+ASG_ARGS。

### 6.（问题3）AGENTS.md 补记 + 纠错
- 追加 asgplus2 isolate 工作的修改记录：sharpen 独立光栅分支、enhanced 分支 means2D/opacity/scales/rotations 全 detach、`get_enhanced_illumination` view_dirs detach、densification 统计自此只来自主 R×L 分支、新旧 run 训练动力学不可比的注意事项。
- 用 python 字节级替换修正上轮 wandb 记录中"恒为 0"的错误理由（文件混合编码，须按 bytes 操作）。

### 7.（问题7 小项）timing_stats 落日志
`train.py:1585` 训练循环结束后补一行 `logger.info("Timing stats: %s", timing_stats)`——目前只累计从不输出；消融比时间正好需要。

### 8.（问题7）StableSR 子模块
执行阶段用 `git --git-dir` 方式查看其未提交改动内容并在总结中报告（计划模式的工具过滤拦截了 `git -C`，只读查看本身没问题）。不代为提交。

## 二、消融实验整理（问题4 + 问题2/3 的量化）

新建 `docs/ablation_reflectance_sharpen_20260928.md` + `scripts/run_ablation_sharpen.sh` + `scripts/render_ablation_sharpen.sh`（复用 run_ablation_9.sh 的 COMMON_ARGS+ASG_ARGS、磁盘检查、skip-existing、端口选取模式；render 脚本沿用 render_ablation_9.sh 的 `--profile_render_timing`，结尾参考 collect_ablation_results.py 输出汇总表）。

九个臂（chair，8k，单 dense 配置）：
- **A0 baseline**：当前全量默认。
- **A1 no_contrast**：`contrast_reg 0`（假设与 edge_uplift 重叠）。
- **A2 minimal_structure**：`contrast 0, edge 0, highfreq 0` 只留 edge_uplift（假设单边 hinge 是唯一非冗余锐化项）。
- **A3 no_highfreq**：`highfreq 0`（与 A1 对照区分 highfreq/contrast 各自贡献）。
- **A4 no_structure_all**：四个结构项全 0（测量整个锐化簇的净贡献）。
- **A5 no_highlight**：`highlight_reflectance_reg 0`。
- **A6 no_tiny_regs**：`consistency 0 + detail 0 + decoder_reg 0`（三个最弱项，假设无可测效应）。
- **A7 isolate_off**：全量正则 + `--disable_reflectance_grad_isolation`（量化隔离分支 ~15-25% 开销换来的质量差）。
- **A8 asg_lobes_2**：`asg_lobes 2`（验证 sum 修复后的多 lobe 行为，探索性，可后跑）。

判读标准写入文档：以 enhanced_gt 的 PSNR/SSIM/LPIPS 为主（lowlight 已 40+ 近饱和）、去掉某项后指标变化在 run 间噪声内即判"可删"；A7 若时间开销 >15% 而 enhanced_gt 差距 <0.1dB 则默认改回非隔离；训练/渲染时间用 timing_stats 日志与 `--profile_render_timing` 对比。

## 三、验证
- `python -m py_compile` 全部改动文件；`pytest tests/` 全绿（现有 10 个 + 新增 sg_utils、backfill 测试）。
- README/AGENTS.md 改动为纯文本，逐条核对。

## 不做（记录为后续）
- train.py `training()` 千行函数拆分、renderer/losses 全量测试补齐（工程量大，单独任务）。
- AGENTS.md 乱码历史段（不可恢复，仅新增记录避开）。