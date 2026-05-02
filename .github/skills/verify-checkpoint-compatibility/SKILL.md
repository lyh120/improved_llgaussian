---
name: verify-checkpoint-compatibility
description: 'Use when: loading old checkpoints, confirming legacy compatibility mode, migrating models from old to new (B0+SG) format, detecting checkpoint version mismatches, validating parameter presence before training resume.'
---

# LL-Gaussian Checkpoint 兼容性验证

验证旧 checkpoint 与当前模型代码的兼容性，检测是否需要进入 legacy 兼容模式，或是否存在参数缺失。

## 工作流程

### 1. **Checkpoint 版本识别**

识别 checkpoint 属于哪个版本：

| 标志 | 版本 | 主要特征 |
|------|------|--------|
| ✅ `base_log_reflectance` in `state_dict` | **新版本 (B0+SG)** | 2026-04-30后的模型，使用 $R = \exp(B_0)$ 参数化 |
| ❌ 无 B0，仅有 `mlp_reflectance` | **旧版本 (Legacy)** | 更新前的模型，使用 MLP 参数化 |
| ⚠️ `illumination_mode` 未定义或为 `legacy` | **兼容模式触发** | 自动进入 legacy_compatibility_mode |

### 2. **参数检查清单** ✓

加载 checkpoint 时验证以下参数：

**核心参数（B0+SG）**
- [ ] `base_log_reflectance` — 每个 anchor 的对数反射率
- [ ] `mlp_sg_illumination` — SG 照度 MLP 权重和偏差
- [ ] `illumination_mode` — 应为 `"sg"`（否则进入 legacy 模式）

**SG 参数**
- [ ] `sg_lobes` — SG 瓣数（通常 4）
- [ ] `sg_directions` — SG 方向参数
- [ ] `sg_sharpenesses` — SG 锐度参数
- [ ] `sg_amplitudes` — SG 振幅参数

**旧版本参数（仅 legacy 模式使用）**
- [ ] `mlp_reflectance` — 旧反射率 MLP（如果存在）
- [ ] `mlp_illumination` — 旧照度 MLP（如果存在）

**Anchor 管理参数**
- [ ] `_anchor_positions` — 锚点位置
- [ ] `_anchor_scales` — 锚点缩放
- [ ] `_anchor_rotations` — 锚点旋转
- [ ] `_num_points_in_batch` — 批次内的点数

### 3. **Optimizer 状态兼容性检查**

Optimizer 状态可能不兼容，需验证：

- [ ] 新参数（如 `base_log_reflectance`）在 optimizer state 中是否有对应的 momentum/variance（Adam 等）
- [ ] 如果新参数缺失 optimizer state，是否会导致学习率异常
- [ ] 是否需要部分初始化 optimizer state

**常见场景**：
```
旧模型 optimizer.state_dict():
  param_groups[0]['params']: [id(mlp_illumination), id(mlp_reflectance), ...]
  
新模型的期望：
  param_groups[0]['params']: [id(mlp_sg_illumination), id(base_log_reflectance), ...]

若 base_log_reflectance 的 state 为空 → 需要手动初始化或从头开始训练该参数
```

### 4. **自动兼容模式检测**

根据 [scene/gaussian_model.py](scene/gaussian_model.py) 的逻辑，以下情况自动进入 `legacy_compatibility_mode`：

- **加载时** (L336, L375, L1008)：
  ```python
  if 'base_log_reflectance' not in state_dict:
      self.illumination_mode = "legacy"
      self.legacy_compatibility_mode = True
  ```

- **推理时** (L1406)：
  ```python
  if self.illumination_mode == "legacy":
      # 使用旧的 mlp_reflectance + mlp_illumination
  ```

**意义**：
- ✅ 旧 checkpoint 可以继续使用，无需重新训练
- ⚠️ 但无法使用新的 B0+SG 特性
- 🔄 若要升级，需要手动在 checkpoint 中添加 B0 参数或从头训练

### 5. **快速检查流程**（2分钟）

```bash
# 方法1：Python 脚本检查
python -c "
import torch
ckpt = torch.load('checkpoint.pth', map_location='cpu')
model_state = ckpt.get('model_state_dict', {})

has_b0 = 'base_log_reflectance' in model_state
has_sg = 'mlp_sg_illumination' in model_state

print(f'Has B0: {has_b0}')
print(f'Has SG Illumination: {has_sg}')
print(f'Mode: {\"new (B0+SG)\" if has_b0 and has_sg else \"legacy\"}')
"

# 方法2：查看 checkpoint 内的 illumination_mode
python -c "
import torch
ckpt = torch.load('checkpoint.pth', map_location='cpu')
gs = ckpt.get('gaussian_state', {})
print(f'Illumination Mode: {gs.get(\"illumination_mode\", \"unknown\")}')
"
```

### 6. **常见问题与解决** 🔧

**问题 1：加载 checkpoint 后模型跑不起来**
- ✓ 首先检查是否进入了 legacy mode（查看训练日志或 render 输出）
- ✓ 确认当前代码是否支持 legacy mode（render.py L23-24 应有兼容代码）
- ✓ 如果日志显示 `using legacy compatibility mode`，则说明 checkpoint 是旧版本

**问题 2：optimizer state 不匹配导致训练异常**
- ✓ 查看 train.py 中是否有 optimizer state 恢复逻辑
- ✓ 新参数建议从初始学习率重新开始优化，而不是继承旧参数的 momentum

**问题 3：想从 legacy checkpoint 升级到 B0+SG**
- ⚠️ 不支持直接转换，建议：
  1. 用旧 checkpoint 在 legacy 模式下推理获得结果（作为参考）
  2. 从头开始用新代码训练新模型
  3. 对比结果质量

**问题 4：多个 checkpoint，不知道哪个是新版哪个是旧版**
- ✓ 使用上述快速检查脚本批量检查所有 checkpoint
- ✓ 按日期和参数特征分类

## 检查清单输出格式

```
## Checkpoint 兼容性检查报告

### 基本信息
- Checkpoint 路径: [path]
- 大小: [size] MB
- 创建日期: [date]

### 版本检测
- [ ] 新版本 (B0+SG) ✅ / 旧版本 (Legacy) ❌ / 混合 ⚠️
- 识别依据: [has B0, has SG, illumination_mode value]

### 参数完整性
- [ ] B0 参数完整 (base_log_reflectance) ✅/❌
- [ ] SG 参数完整 (directions, sharpenesses, amplitudes) ✅/❌
- [ ] Anchor 管理参数完整 ✅/❌
- [ ] 缺失参数列表: [list if any]

### Optimizer 兼容性
- [ ] Optimizer state 与当前参数一致 ✅/⚠️/❌
- [ ] 是否需要部分重新初始化: [yes/no]
- 建议: [reset optimizer / keep as-is / partial init]

### 推荐行动
1. 加载模式: [新版本直接加载 / 进入 legacy 兼容模式 / 需要人工处理]
2. 后续训练: [可恢复训练 / 建议冻结某些参数 / 建议重新初始化]
3. 推理: [完全兼容 / 已自动切换为 legacy 模式 / 可能出错]

### 详细日志
[error messages if any]
```

## 相关文件

- 兼容模式触发点: [scene/gaussian_model.py#L336-L376](scene/gaussian_model.py#L336-L376)
- Optimizer 恢复: [scene/gaussian_model.py#L1400-L1510](scene/gaussian_model.py#L1400-L1510)
- 推理时兼容: [gaussian_renderer/__init__.py](gaussian_renderer/__init__.py) (搜索 legacy_compatibility_mode)
- Checkpoint 保存: [train.py](train.py) (搜索 save_checkpoint)
