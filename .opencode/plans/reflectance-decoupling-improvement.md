# Reflectance 解耦改进方案

## 问题诊断

当前 B0 (`_base_log_reflectance`) 作为裸 `nn.Parameter`，每个 anchor 完全独立，缺少跨 anchor 空间平滑约束。相比作者原方法用 `mlp_reflectance` MLP 预测（MLP 本身有隐式平滑归纳偏置），B0 方案导致：

1. **B0 初始化为全零** (`gaussian_model.py:713`)，`R = exp(0) = 1.0`，无空间先验
2. **B0 学习率偏高** (`gaussian_model.py:749`)，与 `anchor_feat` 相同 `feature_lr=0.0075`，对需要平滑的参数来说过大
3. **反射率正则极弱**：`L_reflectance_consistency` 权重仅 `1e-4`，`L_reflectance_smooth` 仅 `5e-4`
4. **主损失梯度直接流入 B0**：`image_tmp = R * L + residual` 的 L1+SSIM 梯度无任何阻隔
5. **Residual 后期约束太弱**：`L_residual_reg` 从 2.0 衰减到 0.5
6. **`L_diff` 第二项不 detach reflectance** (`train.py:402`)：StableSR 伪 GT 伪影梯度直接注入 B0

---

## 改进一：强化 R 正则

### 1a. 增大反射率正则权重 + 降低 B0 学习率 + 新增参数

**文件：`arguments/__init__.py`**

```python
# ModelParams 中，将：
self.sg_energy_reg = 1e-4
self.sg_smooth_reg = 1e-4
self.reflectance_consistency_reg = 1e-4

# 改为：
self.sg_energy_reg = 1e-4
self.sg_smooth_reg = 1e-4
self.reflectance_consistency_reg = 5e-3    # 从 1e-4 提升到 5e-3
self.reflectance_smooth_reg = 5e-3         # 新增：显式控制 reflectance smooth 权重
self.b0_lr = 0.001                          # 新增：B0 独立学习率（原跟 feature_lr=0.0075）
self.b0_spatial_smooth_reg = 1e-3           # 新增：3D 空间 B0 KNN 平滑权重
```

同时更新 `_backfill_model_compatibility` 的 defaults：

```python
defaults = {
    ...
    "reflectance_consistency_reg": 5e-3,
    "reflectance_smooth_reg": 5e-3,
    "b0_lr": 0.001,
    "b0_spatial_smooth_reg": 1e-3,
}
```

### 1b. B0 使用独立学习率

**文件：`scene/gaussian_model.py`** — `training_setup` 方法中

```python
# 在所有包含 base_log_reflectance 的 param group 中，将：
{'params': [self._base_log_reflectance], 'lr': training_args.feature_lr, "name": "base_log_reflectance"},

# 改为：
{'params': [self._base_log_reflectance], 'lr': training_args.b0_lr, "name": "base_log_reflectance"},
```

涉及行号：749, 766, 782（三个分支都需修改）

新增 B0 学习率调度器（在 `training_setup` 中）：

```python
self.b0_scheduler_args = get_expon_lr_func(
    lr_init=training_args.b0_lr,
    lr_final=training_args.b0_lr * 0.01,
    lr_delay_mult=0.01,
    max_steps=training_args.position_lr_max_steps,
)
```

在 `update_learning_rate` 中新增：

```python
if param_group["name"] == "base_log_reflectance":
    lr = self.b0_scheduler_args(iteration)
    param_group['lr'] = lr
```

### 1c. 新增 3D 空间 B0 KNN 平滑损失

**文件：`utils/loss_utils.py`** — 新增函数

```python
def L_B0_Spatial_Smooth(base_log_reflectance, anchor_positions, knn=8):
    """3D KNN spatial smoothness for B0 (base_log_reflectance).
    Encourages nearby anchors in 3D space to have similar B0 values.
    Uses pairwise distance weighting so closer anchors have stronger constraint.
    """
    from simple_knn._C import distCUDA2
    N = anchor_positions.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=anchor_positions.device)

    # Compute KNN distances using existing distCUDA2
    dist2 = torch.clamp_min(distCUDA2(anchor_positions).float().cuda(), 1e-10)

    # For each anchor, find KNN neighbors via pairwise distance
    # Efficient: use batched approach with anchor positions
    # Approximate: use distance-weighted smoothness
    # For simplicity, use gradient-based approach: penalize B0 gradient weighted by proximity

    # More practical approach: use the existing 1-NN distance as a scale,
    # then compute spatial gradient loss
    # Sort anchors by spatial proximity isn't feasible, so we use a differentiable
    # approximation: random sampling of nearby pairs

    # Simple but effective: for each anchor, penalize difference between its B0
    # and the mean B0 of anchors within a radius. Use the existing KNN from simple_knn.
    # Since we only have 1-NN from distCUDA2, we do a sampling-based approach.

    # Random pair sampling for efficiency
    num_pairs = min(N * knn, 100000)
    idx_i = torch.randint(0, N, (num_pairs,), device=anchor_positions.device)
    idx_j = torch.randint(0, N, (num_pairs,), device=anchor_positions.device)

    pos_i = anchor_positions[idx_i]
    pos_j = anchor_positions[idx_j]
    b0_i = base_log_reflectance[idx_i]
    b0_j = base_log_reflectance[idx_j]

    dist_sq = ((pos_i - pos_j) ** 2).sum(dim=-1, keepdim=True)

    # Use median KNN distance as scale
    median_dist = dist2.median().clamp(min=1e-6)
    weight = torch.exp(-dist_sq / (2 * median_dist ** 2))

    diff = (b0_i - b0_j) ** 2
    loss = (weight * diff).sum() / (weight.sum() + 1e-8)
    return loss
```

**文件：`train.py`** — 训练循环中增加 B0 空间平滑损失

在 `L_reflectance_consistency = L_Reflectance_Consistency(reflectance_image)` 之后添加：

```python
L_b0_spatial_smooth = L_B0_Spatial_Smooth(
    gaussians._base_log_reflectance,
    gaussians.get_anchor,
    knn=8
)
```

在 loss 累加中：

```python
# warmup 分支：
loss += dataset.b0_spatial_smooth_reg * L_b0_spatial_smooth

# normal 分支 (iteration >= opt.update_from)：
loss += dataset.b0_spatial_smooth_reg * L_b0_spatial_smooth
```

### 1d. 增大 reflectance smooth 硬编码权重

**文件：`train.py`**

将 `L_reflectance_smooth = L_Reflectance_Smooth(reflectance_image, illumination_image) * 5e-4` 中的硬编码权重改为使用参数：

```python
L_reflectance_smooth = L_Reflectance_Smooth(reflectance_image, illumination_image) * dataset.reflectance_smooth_reg
```

---

## 改进二：梯度流隔离

### 2a. 主重建 L1+SSIM 中对 reflectance 做 detach

**文件：`train.py`**

核心思路：主重建损失 (L1 + SSIM) 只优化 illumination 和 residual，让 reflectance 仅通过专门的反射率损失被监督。

```python
# 在 normal 分支（非 warmup）中：
if "render_residual" in render_pkg:
    residual_image = render_pkg["render_residual"]
    # 主重建目标：用 detach 的 reflectance，避免噪声梯度注入 B0
    image_tmp = torch.clamp(reflectance_image.detach() * illumination_image + residual_image, 0.0, 1.0)
else:
    residual_image = torch.zeros_like(gt_image)
    image_tmp = torch.clamp(reflectance_image.detach() * illumination_image, 0.0, 1.0)
```

对于 **warmup 分支**保持不变（warmup 没有 residual，R*L 的分解更简单）。

### 2b. L_diff 中 reflectance detach 修正

**文件：`train.py`** — 行 402

当前代码：
```python
L_diff = torch.abs(illumination_enhanced_image * reflectance_image.detach() - refined_image_dict[...].cuda()).mean() \
       + torch.abs(illumination_enhanced_image.detach() * reflectance_image - refined_image_dict[...].cuda()).mean() * 0.2
```

第二项 `reflectance_image` 未 detach，StableSR 伪 GT 的噪声梯度会直接流入 B0。修正为：

```python
L_diff = torch.abs(illumination_enhanced_image * reflectance_image.detach() - refined_image_dict[...].cuda()).mean() \
       + torch.abs(illumination_enhanced_image.detach() * reflectance_image.detach() - refined_image_dict[...].cuda()).mean() * 0.2
```

**原理**：`L_diff` 的目的是让增强输出匹配 StableSR 伪 GT。StableSR 本身有伪影/噪声，不应直接监督 B0。反射率应由专门的 `L_reflectance_consistency`、`L_reflectance_smooth`、`L_b0_spatial_smooth` 和 `L_Illu` 间接约束。

### 2c. 补充：保留一条 reflectance 监督路径

由于 2a 中 detach 了 reflectance，B0 失去了通过主重建损失的监督。需要确保专门的反射率损失足够强。这已在改进一中解决（增大权重 + 3D 空间平滑）。

但为了保证 R 和 L 的乘积仍然能正确重建图像，新增一个间接约束：**反射率结构损失**，通过低通滤波后的 reflectance 参与主重建：

```python
# 在 train.py 中，新增一个弱监督项：低频 reflectance 参与 L1
# 这保证了 R 的整体亮度水平正确，但高频噪声被抑制
from torch.nn.functional import avg_pool2d

reflectance_lowfreq = reflectance_image
# 使用 average pooling 获取低频分量
if reflectance_lowfreq.shape[1] > 1 and reflectance_lowfreq.shape[2] > 1:
    r_pad = reflectance_lowfreq.unsqueeze(0)
    r_low = avg_pool2d(r_pad, kernel_size=3, stride=1, padding=1).squeeze(0)
else:
    r_low = reflectance_lowfreq

image_lowfreq = torch.clamp(r_low * illumination_image + residual_image, 0.0, 1.0)
L_reflectance_lowfreq = torch.abs(image_lowfreq - gt_image).mean() * 0.1
loss += L_reflectance_lowfreq
```

---

## 改进三：B0 初始化改进

### 3a. 从训练图像做简单 Retinex 分解获取初始 B0

**文件：`scene/gaussian_model.py`** — `create_from_pcd` 方法

当前初始化（行 713）：
```python
base_log_reflectance = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
```

改为利用首帧图像做简单 Retinex 初始化：

```python
def _estimate_initial_b0(self, cameras, anchor_positions):
    """Estimate initial B0 from the first training image via simple Retinex decomposition.
    
    Strategy: For each anchor, project to the first camera, look up the pixel value,
    estimate illumination as max-channel, and derive reflectance = pixel / illumination.
    Then B0 = log(reflectance).
    """
    if len(cameras) == 0:
        return torch.zeros((anchor_positions.shape[0], 3), dtype=torch.float, device="cuda")
    
    cam = cameras[0]
    image = cam.original_image.cuda()  # [3, H, W], already normalized [0, 1]
    H, W = image.shape[1], image.shape[2]
    
    # Simple Retinex: illumination ≈ max channel
    illumination_map = image.max(dim=0, keepdim=True)[0].clamp(min=0.1)  # [1, H, W]
    reflectance_map = (image / illumination_map).clamp(0.05, 1.0)  # [3, H, W]
    log_reflectance_map = torch.log(reflectance_map)  # [3, H, W]
    
    # Project each anchor to the image plane
    R = torch.tensor(cam.R, device=anchor_positions.device, dtype=torch.float)
    T = torch.tensor(cam.T, device=anchor_positions.device, dtype=torch.float)
    xyz_cam = anchor_positions @ R + T[None, :]
    
    z = xyz_cam[:, 2].clamp(min=0.001)
    x = (xyz_cam[:, 0] / z * cam.focal_x + cam.image_width / 2.0).long()
    y = (xyz_cam[:, 1] / z * cam.focal_y + cam.image_height / 2.0).long()
    
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z > 0.2)
    
    base_log_reflectance = torch.zeros((anchor_positions.shape[0], 3), dtype=torch.float, device="cuda")
    
    if valid.any():
        valid_x = x[valid].clamp(0, W - 1)
        valid_y = y[valid].clamp(0, H - 1)
        base_log_reflectance[valid] = log_reflectance_map[:, valid_y, valid_x].T
    
    # For invalid points (behind camera), use global average
    if (~valid).any():
        global_mean = log_reflectance_map.mean(dim=(1, 2))  # [3]
        base_log_reflectance[~valid] = global_mean
    
    return base_log_reflectance
```

然后在 `create_from_pcd` 中调用：

```python
# 替换行 713 的：
# base_log_reflectance = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
# 为：
base_log_reflectance = self._estimate_initial_b0(cameras, fused_point_cloud)
```

### 3b. Anchor growing 时 B0 继承改进

**文件：`scene/gaussian_model.py`** — `anchor_growing` 方法

当前行 1257-1264：新 anchor 的 B0 初始化使用了 `scatter_max`，这会导致新 anchor 的 B0 取邻居中最大值，而非平滑均值。改为加权均值：

```python
# 替换：
new_base_log_reflectance = torch.zeros((candidate_anchor.shape[0], 3), dtype=torch.float, device="cuda")
if candidate_mask.any():
    repeated_b0 = self._base_log_reflectance.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, 3])[candidate_mask]
    new_base_log_reflectance = scatter_max(
        repeated_b0,
        inverse_indices.unsqueeze(1).expand(-1, repeated_b0.size(1)),
        dim=0,
    )[0][remove_duplicates]

# 改为：
new_base_log_reflectance = torch.zeros((candidate_anchor.shape[0], 3), dtype=torch.float, device="cuda")
if candidate_mask.any():
    repeated_b0 = self._base_log_reflectance.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, 3])[candidate_mask]
    # Use scatter_mean for smoother B0 initialization
    from torch_scatter import scatter_mean
    new_base_log_reflectance = scatter_mean(
        repeated_b0,
        inverse_indices.unsqueeze(1).expand(-1, repeated_b0.size(1)),
        dim=0,
    )[remove_duplicates]
```

注意：需要确认 `torch_scatter` 已安装（项目中已在其他地方使用 `scatter_max`）。

---

## 改进四（附赠）：Residual 约束增强

### 4a. 提高 residual_reg 下限

**文件：`train.py`** — `LinearDecayWeight` 调用

当前行 212：
```python
weight_scheduler = LinearDecayWeight(initial_weight=2, final_weight=0.5, total_steps=opt.iterations)
```

改为：
```python
weight_scheduler = LinearDecayWeight(initial_weight=2, final_weight=1.5, total_steps=opt.iterations)
```

将下限从 0.5 提高到 1.5，防止 residual 后期过度吸收本该属于 R 的信号。

---

## 涉及文件汇总

| 文件 | 改动内容 |
|------|----------|
| `arguments/__init__.py` | 新增参数 `reflectance_smooth_reg`, `b0_lr`, `b0_spatial_smooth_reg`；增大 `reflectance_consistency_reg`；更新 `_backfill_model_compatibility` |
| `scene/gaussian_model.py` | B0 独立学习率 + 调度器；新增 `_estimate_initial_b0` 方法；`create_from_pcd` 调用初始化；`anchor_growing` 用 `scatter_mean` 替代 `scatter_max`；`update_learning_rate` 增加 B0 LR 更新 |
| `train.py` | reflectance detach（主重建 + L_diff）；L_reflectance_smooth 使用参数权重；新增 L_b0_spatial_smooth 和 L_reflectance_lowfreq；residual_reg 下限提高 |
| `utils/loss_utils.py` | 新增 `L_B0_Spatial_Smooth` 函数 |

---

## 建议的训练命令

```bash
python train.py --eval \
  -s /home/liuyuhao/ll_further/LL-Gaussian/dataset/LLRS-sRGB/chair \
  -m outputs/chair_sg_exp3_b0_improved \
  --gpu 0 \
  --use_sg_illumination --illumination_mode sg \
  --use_3D_filter --use_residual --use_wandb0000 \
  --iterations 8000 \
  --save_iterations 8000 --test_iterations 5000 8000 \
  --position_lr_max_steps 8000 --offset_lr_max_steps 8000
```

新的默认参数会自动生效，也可以通过命令行覆盖：
- `--reflectance_consistency_reg 5e-3`
- `--reflectance_smooth_reg 5e-3`
- `--b0_lr 0.001`
- `--b0_spatial_smooth_reg 1e-3`
