---
name: validate-sg-training-config
description: 'Use when: starting new SG training, migrating from legacy to B0+SG mode, tuning regularization weights, checking parameter initialization, validating illumination mode settings, debugging training instability.'
---

# LL-Gaussian SG 训练配置验证

在启动 B0+SG 训练前验证参数组合是否合理，检查正则项系数的平衡，确保 illumination_mode 和相关参数的一致性。

## 工作流程

### 1. **参数组合预检** ✓

**必需参数（B0+SG 模式）**
- [ ] `--use_sg_illumination True` — 启用 SG 照度学习
- [ ] `--illumination_mode sg` — 设置为 SG 模式（不是 legacy）
- [ ] `--use_3D_filter [True/False]` — 3D 滤波开关（根据数据集选择）

**SG 相关参数**
- [ ] `--sg_lobes [2-8]` — 推荐值 **4**（瓣数越多计算越贵）
- [ ] `--sg_lambda_min [0.5-2.0]` — 推荐值 **1.0**（SG 锐度下界）
  - 过小：SG 过于锐锐，难以优化
  - 过大：SG 过于宽松，表达能力弱

**B0 反射率参数**
- [ ] 无需显式设置，自动在初始化时创建 `base_log_reflectance`
- [ ] 验证点云初始化时是否正确赋予初值

**命令行示例（推荐）**
```bash
python train.py \
  -s ./dataset/LLRS-sRGB/chair \
  -m ./outputs/chair_sg_b0_test \
  --eval \
  --gpu 0 \
  --use_sg_illumination \
  --illumination_mode sg \
  --use_3D_filter
```

### 2. **正则项系数平衡检查** ⚖️

关键参数：定义在 [arguments/__init__.py](arguments/__init__.py)

| 参数 | 推荐值 | 范围 | 作用 |
|------|--------|------|------|
| `sg_energy_reg` | 1e-4 | 1e-5 ~ 1e-3 | 约束 SG 能量分布 |
| `sg_smooth_reg` | 1e-4 | 1e-5 ~ 1e-3 | 约束 SG 锐度变化 |
| `reflectance_consistency_reg` | 0.01 | 0.001 ~ 0.1 | 约束 E·R 一致性 |

**数值级别检查**

```python
# 伪代码：验证权重是否在同一级别
sg_energy = 1e-4       # ✓ 与 sg_smooth_reg 同级
sg_smooth = 1e-4       # ✓ 与 sg_energy 同级
refl_cons = 0.01       # ⚠️ 比前两者大100倍，需留意

# 问题：如果 refl_cons = 1.0，可能完全主导梯度
# 解决：逐步从 0.01 或 0.001 开始，根据日志调整
```

**平衡检查清单**
- [ ] 所有正则项系数 > 0（不禁用任何项）
- [ ] `sg_energy_reg` ≈ `sg_smooth_reg` （通常相等）
- [ ] `reflectance_consistency_reg` ≤ 0.1 （否则可能过度约束）
- [ ] 如果启用了其他损失（`L_Depth_Smooth` 等），检查它们之间的权重平衡

**典型场景**
```
场景1：从 legacy 升级到 B0+SG
→ 建议从低系数开始：sg_energy_reg=1e-5, sg_smooth_reg=1e-5, reflectance_consistency_reg=0.001
→ 逐步增大，观察loss曲线和推理质量

场景2：训练发散（loss → NaN）
→ 可能是反射率一致性项过强
→ 先降低 reflectance_consistency_reg 到 1e-4，重试

场景3：追求更好的低光增强
→ 增加反射率一致性权重（但不超过 0.1）
→ 同时适度增加 SG 正则项，防止过度拟合
```

### 3. **SG 初始化验证**

检查 SG 参数的初始化是否合理：

**初始化源** (scene/gaussian_model.py)

```python
# SG 方向初始化（应该均匀分布在球面上）
self.sg_directions = torch.nn.Parameter(
    get_uniform_points_on_sphere_fibonacci(sg_lobes)
)  # ✓ 使用 Fibonacci 球均匀分布，良好初始化

# SG 锐度初始化
self.sg_sharpenesses = torch.nn.Parameter(
    sg_lambda_min * torch.ones(1, sg_lobes)
)  # ✓ 初始化为下界，给优化空间

# SG 振幅初始化
self.sg_amplitudes = torch.nn.Parameter(
    torch.ones(3, 1, sg_lobes) * 0.5  # RGB 通道，初始为 0.5
)  # ✓ 合理初值，待优化
```

**验证清单**
- [ ] `sg_directions` 是否在单位球面上均匀分布（使用 Fibonacci 采样）
- [ ] `sg_sharpenesses` 是否都 ≥ `sg_lambda_min`
- [ ] `sg_amplitudes` 值是否在合理范围（0~1）
- [ ] 初始化后的 SG 参数是否可以正确计算渲染结果

### 4. **反射率 B0 初始化检查**

[scene/gaussian_model.py](scene/gaussian_model.py) 在点云初始化时：

```python
# B0 初始化（来自点云颜色）
base_log_reflectance = torch.log(point_cloud_colors.clamp(min=1e-5))
# ✓ 取对数，将 [0~1] 映射到 [-∞~0]，便于学习

# 建议验证：
```

**验证清单**
- [ ] `base_log_reflectance` 的初值范围是否在 [-10, 0]（正常情况）
- [ ] 是否存在 `inf` 或 `nan`（如果颜色中有 0）
- [ ] 是否有梯度流（反向传播）

### 5. **illumination_mode 与兼容性检查**

**模式选择**

| 模式 | 值 | 用途 | 前提 |
|------|-----|------|------|
| **新版本** | `--illumination_mode sg` | 使用 B0+SG，推荐默认 | 无 |
| **兼容模式** | `--illumination_mode legacy` | 加载旧 checkpoint 或调试 | 已有 legacy checkpoint |
| **自动检测** | 无需指定 | 加载 checkpoint 时自动选择 | checkpoint 中有标记 |

**配置检查清单**
- [ ] 若新训练：使用 `--illumination_mode sg` ✓
- [ ] 若继续旧训练：使用 `--illumination_mode legacy` ⚠️（无法享受新特性）
- [ ] 若升级旧模型：建议从头训练，而不是修改 illumination_mode

### 6. **数据集相关参数检查**

取决于数据集类型：

| 参数 | LLRS-sRGB | 自定义低光数据 |
|------|-----------|---------------|
| `--use_3D_filter` | True（推荐） | 建议 True |
| `--undistorted` | False | 按相机标定决定 |
| `--add_reflectance_dist` | False | 若有大的反射率变化，设为 True |
| `--add_illumination_dist` | False | 若有显著光照变化，设为 True |

### 7. **学习率与优化器配置**

SG 参数的学习率（定义在 [scene/gaussian_model.py](scene/gaussian_model.py) 的 `optimizer_groups`）

```python
# SG 相关参数的学习率通常比 Gaussian 参数低 1-10 倍
sg_lr = 1e-4  # 或自动计算，详见代码
base_log_reflectance_lr = 1e-3

# 验证清单
- [ ] sg_directions 的学习率是否被正确设置
- [ ] sg_sharpenesses 的学习率是否过大（可能导致参数爆炸）
- [ ] base_log_reflectance 的学习率是否合理（通常与颜色 lr 接近）
```

### 8. **快速验证脚本**（1分钟）

```bash
# 检查是否正确传入了 SG 参数
python -c "
import sys
sys.path.append('.')
from arguments import ModelParams
from argparse import ArgumentParser

parser = ArgumentParser(description='LL-Gaussian')
mp = ModelParams(parser)

# 模拟命令行参数
args = parser.parse_args([
    '--use_sg_illumination',
    '--illumination_mode', 'sg',
    '--sg_lobes', '4',
    '--sg_lambda_min', '1.0',
])

print(f'use_sg_illumination: {args.use_sg_illumination}')
print(f'illumination_mode: {args.illumination_mode}')
print(f'sg_lobes: {args.sg_lobes}')
print(f'sg_lambda_min: {args.sg_lambda_min}')
"

# 检查正则项是否定义
grep -n "sg_energy_reg\|sg_smooth_reg\|reflectance_consistency_reg" arguments/__init__.py
grep -n "L_SG_Energy\|L_SG_Sharpness\|L_Reflectance_Consistency" train.py
```

### 9. **配置报告模板**

```
## SG 训练配置验证报告

### 基本配置
- illumination_mode: [sg/legacy]
- use_sg_illumination: [True/False]
- use_3D_filter: [True/False]

### SG 参数
- sg_lobes: [value] ✓/⚠️
- sg_lambda_min: [value] ✓/⚠️
- 初始化方式: [Fibonacci球/其他] ✓

### 正则项系数
- sg_energy_reg: [value]
- sg_smooth_reg: [value]
- reflectance_consistency_reg: [value]
- 平衡评估: [均衡/有偏差] ✓/⚠️

### B0 反射率参数
- 初始化范围: [min, max] ✓
- 是否有梯度流: [yes/no] ✓

### 兼容性检查
- 是否与当前代码兼容: [yes/no] ✓
- 是否自动进入 legacy 模式: [yes/no]

### 建议
1. [Action 1]
2. [Action 2]
3. [Action 3]
```

## 常见问题 🔧

**Q: sg_lobes 应该设多少？**
A: 推荐 4。更多瓣数（8）计算更贵但表现力更强；更少瓣数（2）计算快但可能欠拟合。从 4 开始。

**Q: reflectance_consistency_reg 太高导致 loss 爆炸？**
A: 逐步调整：0.1 → 0.01 → 0.001。在 training_report 中监控这一项的具体值。

**Q: 新训练时是否需要指定 illumination_mode？**
A: 需要。新训练必须使用 `--illumination_mode sg`，否则会进入 legacy 模式。

**Q: 能否从 legacy checkpoint 直接升级到 B0+SG？**
A: 不建议。新参数（B0、SG）没有对应的 checkpoint 数据，优化器 state 也不匹配。建议从头训练。

**Q: 如何监控训练中 SG 参数的变化？**
A: 在 train.py 的 training_report 中添加日志打印 sg_stats（已有接口）。

## 相关文件

- 参数定义: [arguments/__init__.py](arguments/__init__.py)
- 模型初始化: [scene/gaussian_model.py#L89-L200](scene/gaussian_model.py#L89-L200)
- Loss 计算: [train.py#L365-L377](train.py#L365-L377)
- SG 工具函数: [utils/sg_utils.py](utils/sg_utils.py)
- 推荐训练命令: [scripts/train.sh](scripts/train.sh)
