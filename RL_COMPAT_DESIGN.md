# R/L 配套约束（实验 profile）

新训练显式使用 `--supervision_profile llgaussian_rl --rl_compat_stage sharp|color|stable`。
`sharp` 为边缘和局部对比，`color` 累加可信低光色度，`stable` 再累加表达稳定项。
`--rl_compat_scale 0` 完全关闭新增约束。默认仍为 `llgaussian`，历史配置缺少 profile 仍恢复 `custom`。

R = clamp(exp(B0 + 1.1*tanh(detail)) * (1 + 0.15*tanh(decoder)), .001, 1)。
L 为每 offset 4 个灰度 SG 的均值，参数 MLP 输入特征后半部分和视角；增强 L 保留原版 RGB MLP。
所有 reference profile 共用前向、LL-Gaussian objective、enhancement MLP、densification 和 checkpoint 语义。

新增项 1000–2000 步线性激活，不修改原版 StableSR/深度/重建/残差监督。
R 辅助图复用数值相同的 renderer，位置/屏幕坐标/opacity/scale/rotation/相机/共享特征冻结。
新增 R 项只累积 B0/detail/decoder 梯度；SG 异常锐度项只累积 SG MLP 梯度。
原版主损失仍拥有原版几何优化权限。不能把辅助项直接加到主损失后统一 backward。

低光训练图固定 3×3 二项式降噪；信号 max RGB 4/255→12/255 线性置信度；
排除邻域饱和 >=250/255、coverage<.95 和两像素边界。MAD/.67448975 估计高通噪声尺度。
log 灰度 Sobel 边缘需大于三倍近似 log 噪声、两次平滑后的梯度方向余弦 >=.8；
5×5 log 灰度局部标准差也需超过三倍 log 噪声。启发式可信度不是噪声或反射率真值。
结构损失为 relu(.8t-u)+relu(u-1.25t)，权重 edge .001、contrast .0005。
可信色度 X/sum(RGB) 的平均绝对差权重 .001；未使用额外 StableSR 颜色目标。
稳定项：B0 范围 [log(.001),0] 软保护 1e-5；tanh(detail)² 1e-6；tanh(decoder)² 2e-5；
relu(lambda/64-1)² 1e-4。不开启 B0 空间平滑、平均 SG 幅度/锐度惩罚或几何截断。

每模型保存 rl_compat_config.json（版本/权重/阈值）和每100步 rl_compat_trace.jsonl（实际权重、
各项值、可信区域比例、R/SG辅助参数梯度 L1）。训练图目标只读 CPU 缓存，按需移至训练 GPU。
模型本身使用原有完整 PLY + MLP + pose；新增训练 profile 和 stage 写入 cfg_args。

chair 三组均 fresh seed0、无额外 warmup、8000步。正常光 GT 仅用于评估。
新增约束全部关闭的 reference 行为通过 CUDA 梯度/输出对照；真实模型辅助梯度和保存加载独立审核。
主报告为默认50步 test pose；固定 pose、训练入口500步另列。正常增强 LAB 对齐、raw增强、低光分别算三指标。
结果改善与针状伪影/颜色/结构泄漏均需实际图像验证；未达条件的候选不替换默认基线。
