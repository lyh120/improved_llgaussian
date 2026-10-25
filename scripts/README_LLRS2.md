# LLRS2：LOM 格式与 COLMAP 初始化

服务器目标目录：`/home/liuyuhao/ll_further/LL-Gaussian-sg/datasets/LLRS2`。

源数据：`/home/liuyuhao/ll_further/LL-Gaussian/dataset/LLRS-sRGB`，包含 apartment、building、chair、firehydrant、path、pole、staircase、stone。

经用户确认，使用各场景 **全部正常曝光视角** 运行新的 COLMAP SIFT、全匹配和稀疏 SfM，估计相机与初始点云。此设置与原始 LOM_full 的 high 图建几何方式一致，属于使用正常曝光图像的几何初始化，不是仅低光输入的设置。全部视角包含后续留出的测试视角。

不导入源数据的相机、已有点云、神经重建点云、深度图或增强缓存，也不执行稠密重建或模型训练。

## 场景目录

```text
chair/
├── low/                         # 原始 LLRS images 的独立副本
├── high/                        # 对应 gt/images 的独立副本
├── transforms_train.json        # LOM 风格：训练图引用 low
├── transforms_test.json         # LOM 风格：测试目标引用 high
├── transforms_val.json          # 与 LOM 一样，取一个测试视角作验证
├── transforms_all.json          # 全部相机，图像引用 low
├── colmap.db                    # 此次新建的 COLMAP 数据库
├── colmap_sparse/0 -> ../sparse/0
├── colmap_text/                 # 新模型的 TXT 导出
├── sparse/0/                   # 新模型的 BIN 与 points3D.ply
├── points3d.ply -> sparse/0/points3D.ply
├── images -> low               # LL-Gaussian 的低光训练/测试图入口
├── gt/images -> ../high        # LL-Gaussian 的增强评估 GT
├── dataset_manifest.json       # 每张原图的 SHA256、相机数、点数与核验记录
├── colmap_logs/                # 每个 COLMAP 阶段日志
└── _colmap_work/                # SfM 原始模型，便于审计
```

源 LLRS 的 `images` 与 `gt/images` 已位于去畸变后的像素网格，复制时保留文件名、原始像素、编码与分辨率，不重新缩放或提亮。COLMAP 因此拟合共享的 PINHOLE 相机。LOM JSON 中保留相同的内参字段，畸变系数为零。

JSON 使用 OpenGL/NeRF 相机轴：对 COLMAP camera-to-world 的相机 Y/Z 轴翻转；世界坐标、尺度和点云保持不变。构建时检查转换的往返误差。

划分采用当前项目的默认规则：按图像 stem 字典序排列，每 8 个视角留出 1 个。此设置保证 JSON 与本项目 COLMAP 入口选择相同的训练/测试视角；它没有复制原始 LOM 各场景约 10% 的测试名单。验证集取测试集首个视角。这里的“格式兼容 LOM”指文件结构、相机 JSON 字段与 low/high 引用方式，不要求所有数据集具有相同图像分辨率、场景尺度或测试比例。

本项目发现 `sparse/0` 后使用 COLMAP 分支：训练和低光测试读取 `images`，增强指标读取 `gt/images`。读取 LOM JSON 的框架则遵循 JSON 中 train=low、test/val=high 的目标约定。

## 构建与验证

```bash
/home/liuyuhao/anaconda3/envs/llgaussian/bin/python -u /home/liuyuhao/ll_further/LL-Gaussian-sg/datasets/LLRS2/_build/prepare_llrs2_colmap.py \
  --source /home/liuyuhao/ll_further/LL-Gaussian/dataset/LLRS-sRGB \
  --output /home/liuyuhao/ll_further/LL-Gaussian-sg/datasets/LLRS2 \
  --project /home/liuyuhao/ll_further/LL-Gaussian-sg \
  --colmap /home/liuyuhao/anaconda3/envs/cm/bin/colmap \
  --gpu 1 --threads 8 --holdout 8
```

`build_report.json` 汇总各场景结果。只有输入图像全部成功注册到同一个非空稀疏模型后才发布该场景；不会用旧位姿或插值位姿补齐失败视角。低光与正常曝光副本在复制后校验 SHA256；场景清单记录输入数、注册数、缺失名单、点数和坐标转换误差。

服务器实际执行的脚本和命令保存在 `_build/`。GPU 参数仅用于 SIFT 和匹配，bundle adjustment 使用受限数量的 CPU 线程。
