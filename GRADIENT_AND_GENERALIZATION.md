# 梯度权限与泛化修复

本轮参考用户确认的 `E:\NTIRE\LL-Gaussian`，保留显式反射率、SG/ASG 光照、增强及 transient 部件。修改针对错误的优化通路与投影/合成缺陷，没有提高锐化权重或 R 学习率。

## 方法：符号与含义

显式反射率的参数化为：

\[
R_{a,k}=\operatorname{clip}\left[\exp\left(B_a+\rho\tanh d_{a,k}\right)
\left(1+\eta\tanh D_\theta(\operatorname{sg}(f_a),\operatorname{sg}(o_{a,k}))\right)\right].
\]

`sg` 表示停止梯度。B 是 anchor 的对数底色；d 是 offset 细节；decoder 在固定的局部特征与几何描述上学习外观修正。显式 R 使用原始 anchor 特征，避免 feature bank 的视角调制进入 R。共享特征仍从主重建及原有几何/光照路径学习。

\(\rho\) 是**对数反射率细节的幅度上界**。裁剪前，detail 对底色的乘性变化在 \([\exp(-\rho),\exp(\rho)]\) 内。\(\eta\) 是 **decoder 的最大相对修正幅度**，对应因子范围 \([1-\eta,1+\eta]\)，要求非负且小于单位幅度以保持修正因子为正。它们定义模型表达范围，是有物理解释的约束，不能宣称为经过标定的材质常数。sRGB 输入下的 R 也不能直接当作真实物理 albedo。

每个 Gaussian 先构造低光或增强辐射，再进行一次 alpha 合成：

\[
\hat I=\mathcal S_G(RL),\qquad
\hat I^+=\mathcal S_{\operatorname{sg}(G)}(\operatorname{sg}(R)L^+).
\]

分别光栅化后的 R、L 图仅作诊断；将它们相乘会重复引入覆盖率，并在多个 Gaussian 重叠处引入分量间交叉项，不能代替正式辐射合成。训练、验证、渲染和 fast 增强路径采用一致的辐射定义。

三项锐化使用 \(\mathcal S_{\operatorname{sg}(G)}(R)\)，其中 G 包括中心位置、屏幕坐标、透明度、尺度、旋转及相机变换。它们分别比较灰度梯度、局部标准差和 Laplacian 的归一化响应，以单边 hinge 补足欠缺结构。目标保留比例 \(\tau\) 表示最低期望结构强度；mask 阈值 \(t\) 相对于当前图像的平均响应定义；局部对比窗口定义观察纹理的空间支持范围。方法中使用这些符号及含义，实际取值列入实验设置。

B0 初始化使用多训练视角中有效投影处的 RGB 相对最大通道值，按信号强度在 log 空间加权平均。修正相机矩阵乘法与图像纵轴约定，剔除画面外、相机后方及无信号观测，去掉第一帧的固定曝光下限和人工高频注入。该初始化选择 R/L 的尺度约定，并不提供真实反射率标签。只检查视锥，没有遮挡可见性检验；初始化仍可能受遮挡影响，后续需靠多视角重建纠正。

启用 3D filter 时，采样 footprint 作为世界空间标准差加入**解码后的三轴协方差**。透明度按滤波前后协方差行列式的平方根比补偿，以保持单个 Gaussian 的积分量。如此 decoder 无法将最终尺度缩回采样下限以下，且 offset 中心布局不被滤波移动。普通和 fast 路径使用同样的规则，footprint 按 anchor/offset 顺序匹配。

## 各项损失的权限

| 监督 | 更新对象 | 几何和增密梯度 |
| --- | --- | --- |
| 默认真实低光 L1 / SSIM | 主几何、透明度、R/L 与共享特征，延续主重建路径 | 保留 |
| 可选结构实验：全分辨率真实低光 L1 / SSIM | 仅 R/L 外观参数，使用冻结几何的联合辐射渲染 | 冻结，不进入增密统计 |
| 可选结构实验：5×5 局部平均的真实低光 L1 / SSIM | 主几何、透明度与协方差共享特征；颜色常量化 | 保留，避免逐像素噪声直接驱动几何 |
| 原有 illumination 图像先验与边缘感知平滑 | 仅 SG/ASG/MLP-L；固定输入特征、方向、距离及 rasterizer 几何 | 冻结，不进入增密统计 |
| Depth Anything V2 相对深度约束 | 主几何及遮挡/覆盖相关参数，保留参考项目的局部 PCC 实现 | 保留 |
| R 补边、局部对比、高频三项锐化 | B0、detail、decoder；MLP 模式为 MLP-R | 冻结，不进入增密统计 |
| R 一致性、平滑、亮区约束与参数正则 | 同上 | 冻结 |
| 增强曝光、颜色、平滑、结构与伪 GT 监督 | enhancement SG 与 context 参数 | 冻结；R、L、共享特征也不更新 |
| 可选增强目标对 R 的修正 | 仅 R 参数 | 冻结，并停止增强光照梯度 |
| SG/ASG 能量、锐度、各向异性先验 | 对应 SG MLP 或 ASG 参数 | 参数组限定，不更新 anchor/shared feature |
| transient 重建与正则 | transient 自己的特征、网络与局部几何 | 不修改主 anchor 或相机位姿 |

实现采用冻结渲染输入与参数组白名单两层限制。将整张 R 图 `detach()` 会让 R 自己也无法学习；只用普通 R 分量图则会泄漏到透明度和协方差。主重建执行普通 `backward()`，辅助项只累计对应参数梯度；日志总损失仍包含所有项。

## 实验配置与兼容性

下列数字只记录保留的实验默认值，不是方法成立的前提：

| 配置 | 默认值 | 意义 |
| --- | --- | --- |
| `reflectance_detail_scale` | 1.1 | log-detail 半径 \(\rho\) |
| `reflectance_decoder_scale` | 0.15 | 相对修正上界 \(\eta\) |
| `reflectance_edge_target_ratio` | 0.85 | 最低归一化梯度结构比例 |
| `reflectance_contrast_target_ratio` | 0.8 | 最低归一化局部对比比例 |
| `reflectance_highfreq_target_ratio` | 0.85 | 最低归一化高频结构比例 |
| `reflectance_edge_threshold` | 0.15 | 梯度结构 mask 相对阈值 |
| `reflectance_contrast_threshold` | 0.1 | 局部对比 mask 相对阈值 |
| `reflectance_highfreq_threshold` | 0.1 | 高频结构 mask 相对阈值 |
| `reflectance_contrast_kernel_size` | 5 | 局部对比窗口的像素边长 |

这些参数支持命令行与旧 cfg_args 默认回填。表达幅度参数写入 split 的 `reflectance_config.pth` 和 unite checkpoint，使训练与渲染的表达范围一致。旧模型缺少新配置文件时使用 cfg/defaults。PLY 字段、已有 optimizer 参数组及 capture tuple 布局保持兼容。

单 residual 分支也遵守启用迭代和线性爬坡日程，幅度正则使用绝对值。启用迭代表示允许 transient 参与的阶段，爬坡时长表示其重建权限从零到全量的过渡。原有损失权重、学习率、滤波 footprint 系数与亮区启发式也属于实现/实验配置，不能宣称为通用物理常数。

WandB 每个训练阶段保留总损失、低光 L1、R 辅助损失、增强损失、anchor 数、使用的 residual 幅度。仅保留一个固定相机的 GT/重建/增强/R/L 图，以及定期验证 L1/PSNR。删除随机与固定相机重复图、参数分位数和重复 scalar；增密细节留在本地日志。关闭 WandB 时不再对 `None` 调用 `log()`。

## 验证范围

CPU 测试覆盖三项锐化在显式 R / MLP-R 下的梯度权限、主重建梯度保留、单次 alpha 合成、B0 初始化的曝光/投影/无效观测、多 anchor footprint 排列与积分透明度守恒。CUDA smoke test 在具备 PyTorch CUDA 和项目扩展时检查真实 rasterizer 的辅助梯度隔离与主重建梯度。

运行 `python -m unittest discover -s tests -v`。训练使用冻结几何的 R 和 L 辅助光栅化；启用增强目标对 R 的监督时再增加一次。常规推理不请求辅助图。CUDA 测试分别检查显式/MLP-R 锐化，以及 SG/ASG/MLP-L 光照先验的几何隔离和前向数值一致性。

## 2026-10-08 chair 单场景验证

本轮没有再次提高锐化 loss 权重；当前版本原本已有 R 权限隔离，本轮补齐 L 图像先验的同类隔离。继承 LL-Gaussian 的 StableSR-Turbo 4 步、AdaIN 颜色修正与 Depth Anything V2 ViT-L；默认 2D prior 改为 `stablesr`，CIDNet 仍可通过显式参数选择。LITA-GS 仅作为分解与相对深度监督设计的比较参考，没有复制其网络或替换 StableSR。

两轮 chair 各 8k 已完成。默认 50 步 test 位姿优化、论文 LAB 亮度对齐口径：首轮 PSNR/SSIM/LPIPS 为 17.1532/0.241584/0.630176；结构与有限支撑实验为 17.1860/0.248711/0.700524。额外约束没有改善 LPIPS，且两轮仍有明显偏色、伪影与 R/L 纹理串扰，视觉质量验证未通过。因此额外约束保留为可选实验：默认 `geometry_photo_kernel_size=1`、`gaussian_footprint_limit=0`，保持原主重建路径；明确传入 5 和 16 才复现实验分支。核心 R 锐化与 L 图像先验权限隔离仍默认生效。不能将梯度权限正确宣称为解耦质量或图像质量已解决。

首轮 chair 8k 中，实际梯度审计确认 R/L 辅助项没有几何梯度，但首张 test 仍有严重彩色针状结构。该视角 600705 个可见 offset 中，10.73% 的最长/中间尺度比超过 8，最大尺度/采样 footprint 比达到 1338。这说明主重建的噪声拟合与协方差支撑仍需控制，不能把三项锐化隔离当作充分条件。

据此在可选结构实验中进一步分配主重建的梯度权限：全分辨率光度损失只更新 R/L，几何从 5×5 平均后的真实低光结构和原有相对深度损失学习。两个前向都使用同一 Gaussian 辐射及 alpha 合成，冻结只改变梯度，未更改最终预测公式，也没有将 StableSR 或正常光 GT 引入几何目标。此设置是本模型的改进，不能宣称使用与原论文完全相同的训练目标。

此外，解码后的 Gaussian 标准差先限制在 16×各 anchor 的世界空间采样 footprint 内，再执行原有 3D 抗混叠方差与积分透明度补偿。这个约束在训练、普通渲染和 fast 渲染一致生效，是模型的有限支撑范围，不是对已生成图片做清理。16 是本次诊断设置，不是通用物理常数；可用 `--gaussian_footprint_limit 0` 关闭。旧 cfg 缺该字段时回填 0，保留旧模型渲染行为。旧 cfg 的 `geometry_photo_kernel_size` 回填 1；新训练默认 1；结构实验显式使用 5。

DUSt3R 稠密输入必须在深度 warmup 前执行用户选择的 LLGIM prune_ratio。此前 warmup 强制 `prune_ratio=1`，而主阶段直接恢复 warmup PLY，实际从未应用指定裁剪；本轮恢复 warmup 对该参数的遵守。主阶段不会二次裁剪。

验证选用 `R = B0 + offset detail + decoder`、4-lobe SG 标量低光 L、RGB 增强 SG，训练总预算 8k（2k depth warmup + 6k 主阶段）。这是将论文正文的 8k 总预算分配到当前两阶段模型；参考代码 `--warmup --iterations 8000` 则是 2k+8k，二者应明确区分。参考脚本 voxel=0.001、prune_ratio=0.1、offset LR 0.001→0.00001、颜色 decoder LR 0.04→0.00025。论文补充文字给出 decoder 初始 LR 0.4，与发布脚本不一致；本轮跟随发布脚本。实际硬件是 RTX4090，不能声称复现 V100 训练时间。

采用 sorted cameras 每八帧留出一个 test 的现有 LL-Gaussian 数据划分。StableSR 和深度 prior、B0 初始化仅使用训练低光帧；正常光 `gt/images` 只用于评估。`render.py` 默认 50 步 test 位姿优化只使用低光 test GT；另导出固定输入位姿和训练入口 500 步结果。

`utils/evaluation_utils.py` 对完整 saved test PNG 计算 RGB PSNR、SSIM（11×11、σ=1.5）、VGG LPIPS（[0,1]、normalize=False，遵循发布 train.py）。正常光主指标遵循论文补充式 (21)：在 LAB L 上拟合 GT→prediction 的 a,b，以 `(prediction_L-b)/a` 对齐，仅保留预测 a/b 色度；normal GT 不参与模型训练或位姿优化。并保存无亮度对齐的指标，避免对齐掩盖曝光误差。GT 尺寸或视角缺失直接报错，不静默 resize 或减少评估数量。

`metrics_enhanced_gt.json` 为论文 LAB 对齐结果；`metrics_enhanced_gt_raw.json` 为原始增强结果；`metrics_lowlight.json` 为低光重建。aligned 图仅用于评估展示，不能当作部署时直接预测。独立 R/L 分量图是 alpha 合成诊断，不等于真实物理材质/照明 GT。

本机为 CPU PyTorch，不能以单元测试代替真实跨数据集训练。旧 checkpoint 可以加载，但错误几何和旧初始化不会因为加载新代码自动恢复；质量对照应重新训练。修复了可能导致针状表示、暗边与模糊补偿的具体机制，尚未验证“无伪影”或泛化恢复。

对照应固定数据划分、训练预算和已有参数，比较参考 LL-Gaussian、修改前与修改后版本。除增强 PSNR/SSIM/LPIPS 外，还需检查固定视角细边缘、暗区、强高光及新视角；不能仅依据训练损失判断泛化。
