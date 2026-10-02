# Reflectance 锐化正则与梯度隔离消融方案（2026-09-28）

## 背景与目的

当前主路径（ASG illumination + explicit reflectance）作用在反射率 R 上的结构/正则损失有 8 项，
权重跨度 1e-6 ~ 3e-3，其中四个"锐化"项（edge / edge_uplift / contrast / highfreq）语义高度重叠，
历史调参记录显示该多损失平衡对超参敏感（同一权重经历了 5e-3 → 2e-5 的反复横跳）。
同时 2026-09-28 引入的梯度隔离（sharpen 独立光栅分支）在默认配置下全程生效，
每 iteration 额外一次光栅化与一次 decoder/MLP 前向（粗估 +15-25% 训练时间）。

本消融回答三个问题：

1. **A 系列找出锐化簇的最小充分集**——哪些正则可以删掉而不掉指标；
2. **A7 量化梯度隔离的代价/收益**——多花的时间是否换来可测的质量；
3. **A8 验证多 lobe 叠加语义**——`asg_lobes=2` 配合 sum 合并的行为（探索性，可最后跑）。

## 实验设置

- 场景：`chair`（快迭代场景；结论确认后可扩 `buu`/`sofa`，脚本已支持 `--scene`）。
- 训练：8k iterations，单 dense 配置（与 `scripts/run_ablation_9.sh` 的 `COMMON_ARGS` 完全一致），
  ASG 主路径（`ASG_ARGS`），CIDNet 固定 round_000 先验（`--cidnet_refresh_interval 0`）。
- 每臂唯一差异 = 变体参数，其余完全一致，保证单变量归因。
- 输出目录：`outputs/${scene}_sharpen_${variant}_8k`。

## 消融臂

| 臂 | 变体参数 | 假设 |
| --- | --- | --- |
| A0 `a0_baseline` | 无（当前全量默认） | 参照系 |
| A1 `no_contrast` | `--reflectance_contrast_reg 0` | contrast 与 edge_uplift 语义重叠，可删 |
| A2 `minimal_structure` | `--reflectance_contrast_reg 0 --reflectance_edge_reg 0 --reflectance_highfreq_reg 0` | 单边 hinge 的 edge_uplift 是唯一非冗余锐化项 |
| A3 `no_highfreq` | `--reflectance_highfreq_reg 0` | 与 A1 对照，区分 highfreq 与 contrast 各自贡献 |
| A4 `no_structure_all` | 四个结构项全 0 | 测量整个锐化簇的净贡献（上界检查：若不掉指标，整个簇都可删） |
| A5 `no_highlight` | `--highlight_reflectance_reg 0` | 高光色彩异常压制项的必要性 |
| A6 `no_tiny_regs` | `--reflectance_consistency_reg 0 --reflectance_detail_reg 0 --reflectance_decoder_reg 0` | 三个最弱项（2e-5/1e-6/2e-5）无可测效应 |
| A7 `isolate_off` | `--disable_reflectance_grad_isolation` | 隔离分支的时间开销 vs 质量差 |
| A8 `asg_lobes_2` | `--asg_lobes 2` | 多 lobe 叠加（sum）行为验证，探索性 |

运行：

```bash
bash scripts/run_ablation_sharpen.sh --gpu 0 --scene chair
bash scripts/render_ablation_sharpen.sh --scene chair   # 渲染 + 汇总指标表
```

## 判读标准

1. **主指标**：`test/ours_8000/metrics_enhanced_gt.json` 的 PSNR / SSIM / LPIPS。
   低光指标（metrics_lowlight）已 40+ dB 近饱和，只作 sanity check，不作为删留依据。
2. **可删判据**：去掉某正则后 enhanced_gt 指标变化在 run 间噪声内
   （chair 单场景建议以 |ΔPSNR| < 0.15 dB 且 LPIPS 不变差为准；处于边界时重跑一次再定）。
   A2 若不差于 A0，则最小集就是 {edge_uplift}；若 A2 变差但 A1/A3 至少一个不差，按其结果收敛。
3. **A7 判据**：对比 A0 与 A7 的
   - 时间：训练结束日志 `Timing stats (seconds):` 中的 `render_time` / `total_time`
     （除以迭代数得每迭代开销），以及渲染侧 `--profile_render_timing` 的 FPS；
   - 质量：enhanced_gt 指标差。
   若时间开销 > 15% 且 enhanced_gt 差距 < 0.1 dB，则把 `disable_reflectance_grad_isolation`
   的默认值翻转为 True（或至少在追求速度的批量场景脚本中默认关闭）。
4. **A8 判据**：观察 wandb `asg_energy`/`asg_anisotropy` 曲线与 enhanced_gt 指标，
   仅决定是否值得继续投入多 lobe 方向；sum 合并语义本身已由单测锁定。
5. **次要观察**：`reflectance_*` 系列 wandb 标量（确认被删正则确实归零、其余项分布不变），
   以及 `monitor/fixed/reflectance` 图（确认锐化/模糊的视觉差异与指标一致）。

## 成本估计

9 臂 × 8k iters；按单卡单臂约 1-2 小时（含 CIDNet 先验生成）估计，chair 单场景全序列约
1-2 天。A8 探索性可放最后，时间紧张时优先 A0/A1/A2/A7 四臂（已能回答主问题）。

## 结果记录

渲染脚本结尾调用 `scripts/collect_sharpen_ablation_results.py`，输出
`outputs/sharpen_ablation_results[_scene].json`（enhanced_gt PSNR/SSIM/LPIPS + 渲染 FPS），
训练时间从各 run 的 `Timing stats (seconds):` 日志行人工摘录到同一表格。

## 相关代码改动（本方案依赖）

- `--disable_reflectance_grad_isolation`（`arguments/__init__.py` + `train.py`）：A7 臂开关。
- ASG 多 lobe 合并 mean→sum（`utils/sg_utils.py`）：A8 臂前提；默认 `asg_lobes=1` 下无行为变化。
- `train.py` 训练结束打印 `Timing stats (seconds):`：A7 时间对比数据来源。
