---
name: check-loss-design
description: 'Use when: analyzing loss function design in LL-Gaussian, verifying B0+SG loss consistency, checking regularization terms (energy/sharpness/smoothness), validating weight balance, assessing gradient flow, ensuring numerical stability, comparing code implementation with paper design.'
---

# LL-Gaussian Loss Function Design Review

检查 LL-Gaussian 项目中的loss函数设计是否合理的深度分析技能。此技能用于验证loss函数的**数学正确性**、**设计一致性**、**权重平衡**和**实现稳定性**。

## 工作流程

### 1. **识别与上下文收集** 
   - 🎯 找出所有loss函数定义（通常在 `utils/loss_utils.py`）
   - 📍 定位loss在训练循环中的使用位置（通常在 `train.py`）
   - 📖 查阅项目 `AGENTS.md` 的最新设计说明（当前版本：**B0 + SG主路径**）
   - 🔍 识别关键参数：  
     - 照度模式（`illumination_mode`：sg vs legacy）
     - 反射率参数化（`base_log_reflectance` B0 vs 旧的 `mlp_reflectance`）
     - 正则项系数（`sg_energy_reg`, `sg_smooth_reg`, `reflectance_consistency_reg`）

### 2. **数学正确性审查** ✓
   检查以下方面：

   **a) 基础损失（Photometric Loss）**
   - [ ] L1/L2/SSIM 的数学定义与标准一致
   - [ ] 维度验证：`(1.0 - lambda_dssim) * L1 + lambda_dssim * SSIM` 的权重归一性
   - [ ] 梯度流：检查loss对输出的导数是否正确（通过对比 PyTorch 官方实现或论文公式）

   **b) SG（Spherical Gaussian）相关损失**
   - [ ] `L_SG_Energy`: 确保能量项的数学表达式与论文/设计文档一致
   - [ ] `L_SG_Sharpness`: 锐度项的计算是否符合SG性质（e.g., 方差/协方差项）
   - [ ] 单调性检查：这些正则项是否都是"越小越好"的形式

   **c) 反射率一致性（Reflectance Consistency）**
   - [ ] `L_Reflectance_Consistency` 的目标明确性：是否要求 $E \cdot R$ 一致
   - [ ] `L_Reflectance_Smooth` 的平滑化策略：是否使用梯度平滑或拉普拉斯算子
   - [ ] 约束的物理意义：检查是否符合低光增强中的照度-反射率分解原理

### 3. **B0 + SG 设计一致性审查** ✓
   当前官方主路径（2026-04-30）：

   - [ ] **主反射率参数化**：`R = exp(B0)` 是否在loss中被正确使用
     - [ ] 不应再使用旧的 `mlp_reflectance` 作为主路径（仅在legacy兼容模式下）
     - [ ] 检查是否存在过时的注释或无用代码
   
   - [ ] **SG照度表示**：是否完全使用SG参数化（不混合旧的`mlp_illumination`）
     - [ ] loss中获取的`sg_stats`是否来自正确的渲染器版本
     - [ ] 是否存在warmup阶段的不一致切换（2026-04-30后应移除）

   - [ ] **正则项同步**：loss中三个新正则项是否在train.py中被同时激活
     ```
     sg_energy_reg * L_sg_energy
     sg_smooth_reg * L_sg_sharpness  
     reflectance_consistency_reg * (L_reflectance_consistency + L_reflectance_smooth)
     ```

### 4. **权重系数与缩放审查** ⚖️
   
   - [ ] **基础loss权重**：`lambda_dssim ∈ [0, 1]`，推荐范围 0.1~0.4
     - 检查是否硬编码或从命令行可配置
     - 验证注释中是否有指导值
   
   - [ ] **正则项系数**：检查以下参数的合理性
     ```
     sg_energy_reg      # 通常 1e-4 ~ 1e-3
     sg_smooth_reg      # 通常 1e-4 ~ 1e-3
     reflectance_consistency_reg  # 通常 0.01 ~ 0.1
     ```
   - [ ] **缩放一致性**：所有loss项是否在同一数量级（避免某项主导梯度）
   - [ ] **数值范围**：检查loss项的典型范围
     - L1: 0~1（pixel value normalized）
     - SG能量/锐度：是否有非常大或非常小的值

### 5. **正则项合理性审查** 🔧

   **SG Energy 正则**
   - [ ] 目的明确：是否为了约束能量分布平衡
   - [ ] 数学：能量如何定义（积分、采样点求和、norm2等）
   - [ ] 梯度性质：能量项对SG参数的导数是否well-conditioned

   **SG Sharpness 正则**
   - [ ] 是否真的在控制SG的尖锐程度（vs 宽松程度）
   - [ ] 与论文或经验值的对应性

   **Reflectance Smoothness 正则**
   - [ ] 光滑项的数学形式（TV norm / Laplacian / 梯度范数）
   - [ ] 是否正确地抑制了反射率的无谓振荡

### 6. **Numerical Stability 检查** 🛡️

   - [ ] **分母保护**：
     - [ ] `l1_plus_loss` 中 `phi=1e-3` 是否足够大，防止 `/0`
     - [ ] `l2_plus_loss` 中 `weight1 = 1/(network_output + phi)` 是否会产生NaN
   
   - [ ] **指数/对数稳定性**：
     - [ ] 如果loss中使用 `log(x)`，是否有 `log(x + eps)` 的保护
     - [ ] 如果使用 `exp(x)`，是否可能overflow（x > 100）
     - [ ] SG能量计算是否使用了数值稳定的形式
   
   - [ ] **权重锁定**：
     - [ ] `.detach()` 的使用是否正确（权重不应该更新）
     - [ ] 是否误用了 `.detach()`，导致停止梯度流

   - [ ] **Clipping与Normalization**：
     - [ ] loss是否需要clipping（e.g., `torch.clamp(loss, min=0, max=10)` 防止outliers）
     - [ ] 是否需要在batch维度上normalization

### 7. **代码 vs 论文设计对应审查** 📄

   - [ ] 如果有参考论文/设计文档：
     - [ ] 比较公式与代码实现的一一对应关系
     - [ ] 检查是否有设计演进的遗留代码（e.g., 注释中的旧版本）
   
   - [ ] 如果有多个条件分支（illumination_mode等）：
     - [ ] 每个分支的loss是否都经过验证
     - [ ] 是否存在unreachable代码分支

### 8. **梯度流与反向传播审查** 🔙

   - [ ] **Loss → 参数的梯度路径**
     - [ ] 检查是否所有训练参数都能收到gradients
     - [ ] 是否有参数被意外frozen或detached
   
   - [ ] **Loss分解的可追踪性**
     - [ ] 每个loss项是否都被log记录（便于debug）
     - [ ] 在 `training_report` 中是否记录了所有正则项

## 执行检查清单

1. **快速健康检查**（5分钟）
   ```
   ✓ loss_utils.py 中所有loss函数都有清晰定义
   ✓ train.py 中loss的组合方式符合设计
   ✓ 权重系数在合理范围内
   ✓ 无明显的NaN/Inf风险
   ```

2. **深度审查**（15-30分钟）
   ```
   以上 + 
   ✓ 逐个验证每个loss项的数学正确性
   ✓ 检查与B0+SG设计的一致性
   ✓ 分析权重间的平衡关系
   ✓ 追踪梯度流完整性
   ✓ 对比参考文献验证
   ```

## 输出报告模板

```
## Loss 函数设计审查报告

### 1. 审查范围
- 审查文件: [list]
- 当前illumination_mode: [sg/legacy]
- 扫描的loss项: [list]

### 2. ✓ 符合项 (Pass)
- Item1
- Item2

### 3. ⚠️ 需要关注的地方 (Warning)
- Issue1 @ file.py#L123 - 描述 - 建议修复方案

### 4. ❌ 必须修复的问题 (Fail)
- Critical1 @ file.py#L456 - 描述 - 修复代码示例

### 5. 💡 优化建议 (Recommendation)
- Suggestion1
- Suggestion2

### 6. 结论
[Summary with risk assessment]
```

## 示例使用场景

- **场景1**: 新增SG正则项后，验证loss设计是否正确
  > "检查loss_utils.py中的L_SG_Energy和L_SG_Sharpness的实现，验证与B0+SG设计的一致性，并检查train.py中的权重配置是否合理"

- **场景2**: 遇到训练发散或收敛缓慢
  > "深度分析loss函数的梯度流，检查是否存在权重不平衡或numerical stability问题"

- **场景3**: 代码review前的自检
  > "对loss_utils.py和train.py的loss相关部分进行完整的设计审查，生成检查清单"

## 相关资源

- 项目设计文档: [AGENTS.md](AGENTS.md#-2026-0430-b0--sg-正式主路径)
- Loss函数实现: [utils/loss_utils.py](utils/loss_utils.py)
- 训练入口: [train.py](train.py) (搜索 "loss = ")
- 损失计算区间: [train.py#L365-L377](train.py#L365-L377)
