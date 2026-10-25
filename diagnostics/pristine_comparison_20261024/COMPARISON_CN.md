# 纯净 LL-Gaussian 与当前 ASG 项目的代码对比

对比日期：2026-10-24。

参考目录：`E:/NTIRE/LL-Gaussian`。当前目录：`E:/NTIRE/LL-Gaussian-ref-sgs`，分支 `asgplus2`，包含工作区尚未提交的改动。本次只审查源码，未修改训练实现，未启动训练。以下“当前实验”指前面审计过的 LLRS2 firehydrant 6k 实验；它的命令行覆盖值与项目默认值分开说明。

**结论：当前项目不是单独替换 illumination 的原版。** R 表达、基础光照、增强光照、初始化、warmup 恢复、增密裁剪、监督项及部分梯度都发生了变化。相同 StableSR 名称和相同主重建公式，不能保证相同优化过程。源码差异可以定位排查对象，但不能替代同数据、同预算的训练对照来证明伪影根因。

**1. 保留的底层机制**

- 仍然是 anchor + offset 的 Scaffold Gaussian 几何，opacity/covariance MLP 依赖共享特征和视角，3D filter 机制延续原版。
- `scene/cameras.py`、`utils/camera_utils.py`、`utils/pose_utils.py`、`utils/graphics_utils.py` 在归一化换行后相同。
- 三个 rasterization 子模块自身的 Python/C++/CUDA 源文件共 44 个相同，simple-knn 的 7 个对应源文件相同；排除了 build、third_party、缓存和 Git 目录。未核对已编译二进制。
- `configs/config.gin` 相同；并非换了一套 StableSR 采样配置。
- 训练实际使用分别光栅化的分量乘积 `S(R) * S(L)`，增强为 `S(R) * S(L+)`。原版 `gaussian_renderer/__init__.py:475,607` 已经如此。当前 `utils/composition_utils.py` 延续这一公式。
- 加权绝对 L1、SSIM、基础亮度约束和 Gaussian 体积代理正则的主体延续原版。anchor 位置学习率初末默认均为零，也是原版设置；offset 和 covariance 等仍可改变几何。

图像空间乘积一般不等于 `S(R*L)`，但这是共同机制，不能直接作为当前改造特有的退化原因。当前普通 renderer 额外输出的 `render` 字段实际是 `S(R*L)`；训练和正式渲染入口通过 compose helper 使用分量乘积。直接消费原始 `render` 字段的外部代码需要注意语义与原版不同。当前 fast 路径改成两次分量渲染后相乘，以对齐正式输出；旧版单次 fast 渲染的耗时不能直接作新版本 FPS 基准。

**2. 模型表达的三个独立变化**

| 分量 | 纯净版 | 当前 ASG + explicit 路径 |
|---|---|---|
| 反射率 R | 共享 MLP，以 anchor 特征及可选距离为输入，输出每个 offset 的 RGB sigmoid | anchor 的 B0、offset 的 log-detail，再乘共享 decoder 的有界修正 |
| 基础光照 L | 共享 MLP，以后半特征、视角和可选距离为输入，输出标量 sigmoid | 每 anchor/offset 的直接 ASG 参数，含方向、切向、双轴锐度、幅度及基底 |
| 增强光照 L+ | 共享 enhancement MLP，输入 detach 的特征及 illumination logits，输出 RGB sigmoid | 每 offset 的 RGB SG 方向响应，乘上下文门，再加 detach 的基础光照并 clamp |

原版位置：`scene/gaussian_model.py:184-211`、`gaussian_renderer/__init__.py:113-145`。

当前位置：`scene/gaussian_model.py:917-954`、`gaussian_renderer/__init__.py:37-75`、`utils/sg_utils.py:66-132`。

当前 R 的形式为：

`R = clamp(exp(B0 + rho*tanh(detail)) * (1 + eta*tanh(decoder(feature, offset))), floor, 1)`。

这同时改变了跨 anchor 的参数共享、局部自由度、初始化和优化条件。当前 R 参数更多，并不能据此断言结构拟合更容易。主重建仍能更新共享特征和几何，只有三项辅助锐化损失隔离这些输入。原版 R 默认也不直接使用 view direction；不能把“R 不接视角”当作当前版本独有改动。

当前 ASG 是直接逐点学习的光照参数；SG 模式则由 SG MLP 产生参数，两者不能混为同一种实现。当前 L+ 仍是 SG，不能把整个增强路径称为 ASG。当前 L+ 在 Gaussian 层面只能在固定基础 L 之上增亮，原版增强 MLP 没有这个加性下界；它们是不同的表达约束。

即使选择 `illumination_mode=mlp` 和 `reflectance_mode=mlp`，当前增强路径、warmup、增密和额外损失仍不同，因此这两个开关不能完整恢复纯净版训练。

**3. R 初始化是新增的训练假设**

原版不从图像初始化 B0，R 来自随机初始化的共享 MLP。当前 `utils/reflectance_utils.py:29` 用第一张训练图投影到点上，通过 `I/max(max_channel(I), intensity_floor)` 生成 R，再进行局部细节增强。该做法没有多视角融合，也没有遮挡深度测试；仅有投影有效性检查。未命中有效像素的点使用该图有效区域的平均 log-R。

当前强度下限默认 0.1，细节增益默认 0.4。极暗像素低于强度下限时，结果是 `I/0.1`，不等于色度归一化 `I/max_channel(I)`；R 初值会保留输入亮度和噪声的影响。ASG 又以较低基础亮度初始化。这可能影响 R/L 分配及增强学习起点，是应检查的新增机制，但尚未通过单因素训练证明它导致本次失败。

当前 Scene 给初始化传训练相机，原版传测试相机。原版在此使用相机主要用于点云可视化与天空点构造，并没有 B0 图像采样；不能把它解释成原版用测试图初始化 R。切换到训练相机同时可能改变天空点筛选。

**4. warmup 到主训练不是同一种恢复方式**

纯净 `train.py:159` 使用 `only_ply=True`，因此加载 warmup 的 PLY 参数和已有 pose 文件，但跳过 `load_mlp_checkpoints`。新建模型中的 opacity/covariance/R/L/enhancement 等 MLP 从初始化状态开始主训练。PLY 中的 anchor 特征、offset、scale 等继续保留，不能简化为只保留 xyz。

当前 `train.py:455` 使用 `only_ply=False`，加载 warmup 的网络以及扩展后的点参数。两版主训练都重新设置 optimizer；当前这种恢复并不是连同 warmup optimizer 状态连续训练。

两版 warmup 均为 2000 步，并设置较小 offset 初始学习率和较大颜色初始学习率。原版高颜色学习率作用于 MLP；当前 ASG 路径中，同名颜色学习率驱动 ASG 的直接方向、锐度、幅度、基底等参数，数值相同不代表同等优化尺度。当前独立 B0/detail/decoder 又使用各自学习率。

**5. 增密裁剪是另一项实质性变化**

| 行为 | 纯净版 | 当前版 |
|---|---|---|
| 初始点云缩减 | 距离相关概率迭代下采样，prune_ratio 为大致保留目标 | 核心算法保留；实际命令的 prune_ratio=1 跳过此过程 |
| warmup 裁剪 | 删除低累计透明度点及统计期间未见的旧点 | 调用 allow_prune=False，保留初始几何 |
| 增密候选 | 梯度阈值加随机候选筛选 | 梯度阈值加候选 voxel 梯度 Top-K |
| 增密预算 | 没有当前这套总点数/单次/分层硬预算 | 总 anchor 上限、单次上限、分层上限 |
| 主训练裁剪 | 满足累计透明度及观测条件即裁剪 | 增加启动时机、未见点年龄条件和可选单次裁剪预算 |

位置：原版 `scene/gaussian_model.py:1100-1264`；当前同文件 `1984-2095`、`2269-2456`，以及 `train.py:1452`。

当前 firehydrant 实验使用较早的 warmup 增密、更早的 depth 激活，以及主阶段延迟到 4500 步才裁剪、每次最多删除 128 点。观察到 final 74,096 anchors 未达到总数 90,000；这不能排除单次及分层预算一直限制候选。此前末次记录也有大量待裁剪候选。这个差异会影响点的空间分布和形状，不能仅当作显存优化；但更多点或更激进裁剪是否更好仍需对照。

**6. loss：基础公式延续，附加目标与权限变化明显**

纯净版 warmup 主要为：加权 L1 + SSIM + L_Illu + volume；达到阶段阈值后增加 illumination smooth 与 depth。主训练再加 degree、增强平滑、StableSR 双向 detach 监督，以及可选 residual 正则。

当前在此基础上加入 R 的局部一致性、边缘、三项锐化、高亮抑制、detail/decoder/B0 正则和 SG/ASG 参数正则；增强侧增加 gain smooth、edge preserve、颜色均值/标准差、green bias、独立 prior 开始时机、ramp/末期权重以及增强参数梯度裁剪。是否真正生效取决于命令行权重。R consistency 当前是图像空间局部变化约束，不是显式多视角对应一致性。

原版增强像素监督：

`|S(L+) * stopgrad(S(R)) - T| + 0.2 * |stopgrad(S(L+)) * S(R) - T|`。

当前保留该 stop-gradient 结构，但当前 firehydrant 命令将第二项权重设为 0；原版从 `2*update_from` 开始，当前实验从 3500 开始并爬坡。两者给 R 的增强目标不同。当前颜色/保边等附加项又直接使用未 detach 的增强图，因此第二项设为零不代表所有增强监督都不能训练 R。

增强光照 rasterization 在两版都只 detach 了 means3D，并未 detach opacity、scale、rotation 和屏幕张量。像素表达式中的双向 detach 也不等于整个几何冻结；这是原版机制的延续。

三项 R 锐化是当前新增项，已改为辅助 appearance-only 路径；它们不能再通过几何和共享特征降低损失，也不贡献增密屏幕统计。普通 R 边缘、高亮等其他正则仍使用主图，保留其原有权限。此前 firehydrant 单视角 autograd 检查已经验证三项锐化对几何为零，而主重建几何梯度存在。

**7. Depth：监督定义相近，实际梯度不同**

两版均使用训练低光图产生 Depth Anything V2 prior，保持“累积深度取反并归一化 + 局部 Pearson”；当前没有改成用覆盖率除累积深度，也没有自动使用 high 图生成监督。

当前修复 numpy/Tensor、分辨率、无效像素、小图、合法采样边界和退化窗口；原版 `pearson_depth_loss` 将 `1-r` 截到 [0,1]，使负相关区域的纠正梯度被截断，当前允许到 2。

此外当前用辅助 RGB 通道累积相机深度，补齐对中心 z 值的直接梯度。原版直接使用扩展导出的 depth，已有覆盖/投影相关梯度不能等同于完整的深度值梯度。因此保持权重 0.15，不等于保持几何优化强度。此前 firehydrant 的单视角审计中，该 depth 项对 offset/opacity 的梯度与主重建同量级；这不是 depth 有害的证明，而是不能用标量 loss 小来忽略它。

**8. StableSR/CIDNet 的兼容性与相同性边界**

原版只有 StableSR，增益来自训练低光均值，输入为低光乘增益、不做预先 clamp。当前保留 StableSR，也支持 CIDNet，并改进根目录路径、冻结释放、缓存部分补齐、UID/图像名兼容、分辨率及输出范围统一。

当前 StableSR 的默认 input_mode 是 clamped；要对齐原版输入流程，需要显式 `--stablesr_input_mode llgaussian`。此前 firehydrant 实验已经选择该模式。仅 `--enhancement_prior stablesr` 并不自动对齐原版输入。

两种 input_mode 目前使用相同 `diffusion_prior_<gain>` 缓存目录；完整缓存直接复用。修改 input_mode 不会自动重建目标。原版首次生成使用浮点结果，而后续读 PNG；当前始终重新读取保存的 PNG，使首次与缓存运行统一。这是小的量化差异，不能声称逐位等价。

**9. Residual 需要看开关，不能按历史文档推断当前生效路径**

纯净版与当前命令行的 use_residual 默认均为 False。原版启用后，主训练全图加入单 residual，warmup 不加入。

当前 use_residual=True 且 use_dual_transient=False 时，也从主阶段开始全量加入单 residual；只有 dual transient 才使用 noise/artifact 双分支、mask 与 ramp。当前 single 路径没有强制使用旧文档描述的困难区域 mask。

当前 SSIM 与 L1 对同一个完整重建监督；原版 SSIM 忽略 residual，L1 包含 residual。单 residual 末期正则默认当前为 1，原版为 0.5，可通过配置区分。当前 firehydrant 没有启用 residual，所以这些分支不是该次失败的直接运行路径。原版普通最终输出也没有自动加 residual，不宜把默认无 residual 输出说成当前独有发明。

**10. 两个已发现问题属于继承问题**

- warmup 恢复读取 pose 时，load_pose 返回未开启 requires_grad 的张量；主训练没有重新开启，optim_pose=True 仍可能没有真正优化 pose。原版同样如此。测试过的当前 firehydrant 实验确实出现此情况；本次没有运行纯净版训练来核实其具体实验结果。
- training_report 对 test camera 也索引只含 train poses 的 P；train/test UID 分别编号，造成错视角测试。这一代码同样来自纯净版。它污染训练期间的 test 指标，但不是主训练真实视角错位的证据；当前固定监控与独立正确位姿渲染不能一并判错。

另外原版 l1_loss 返回 signed difference，evaluation 对其直接取 mean 的日志问题也被继承；真实训练的 Ll1 明确取了 abs，不能误称真实 L1 没有绝对值。

**对结构失败的排查优先级**

1. 修复/对齐 pose 恢复和评估，先保证测量可信；这不是已证实的全部伪影原因。
2. 对齐 warmup 恢复和点云更新协议，避免同时更换网络表达与几何生成机制而无法归因。
3. 单独验证 explicit R 初始化、直接 ASG、SG 增强网络各自对低光训练拟合的影响。它们是三个独立的模型变化；不能仅开 MLP 模式就声称原版复现。
4. 单独检查 depth 新增的直接几何梯度与增强监督对 opacity/covariance/增密的影响。权限正确不等于所有目标相容。
5. 在主重建能稳定拟合之后，再评价新增的 R 锐化和颜色等正则。

此前当前实验的 27 个训练视角审计显示，anchors 从 22,256 增加到 74,096，而主重建均值从约 0.31380 到 0.31327，存在训练拟合停滞的证据；其中仍可能包含低光噪声造成的不可约误差。尚未完成同数据同协议的纯净版对照，所以不能断言“纯净版必好”或“ASG 必然失败”。本次结果是源码归因地图，不是完成了训练消融验证。
