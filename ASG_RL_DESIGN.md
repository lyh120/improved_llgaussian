# 显式 R / ASG L 与论文配套约束

新训练默认低光 L 为 `asg`，每 offset 一个显式 lobe。主轴与切向量构造正交 x/y/z，
幅度 sigmoid，两个锐度 softplus+1，灰度底光 sigmoid；视角只参与求值，关闭距离衰减。
R 继续为 B0 + 1.1*tanh(detail) 与 1+0.15*tanh(decoder) 的乘积并截到 [.001,1]。
1.1 定义 log 反射率细节范围（约 .33–3 倍），.15 定义 decoder 的 ±15% 修正；
它们是相对材质表达范围，不是物理标定值。R decoder 的特征/offset 输入保持 detach。
增强仍使用原版 LL-Gaussian RGB MLP，特征/低光 L 输入均 detach，先验为 StableSR。

基线 `--supervision_profile llgaussian --illumination_mode asg --asg_lobes 1`。
实验 `--supervision_profile llgaussian_rl --illumination_mode asg --asg_lobes 1 --rl_compat_stage asg_paper`。
`--rl_compat_scale 0` 关闭新增项；旧配置缺少新字段时不启用。旧 SG 阶段/模型继续可用。

R 边缘 .001、局部对比 .0005：目标真实低光经3×3二项式降噪后，在 log 灰度上计算，
限制响应在 [.8t,1.25t]。可靠材质边缘额外要求 RGB 色度 Sobel 超过3倍启发式噪声阈值，
两次平滑后方向余弦≥.8。对比项在5×5材质边缘邻域内启用且通过原可靠对比筛选。
可信平坦区局部色度一致性 .0001：5×5 log灰度/色度标准差均低于3倍噪声阈值，
只约束两端都有效的四邻接色度差。MAD高通残差估计噪声，log按灰度归一，色度按RGB和归一。
阈值下限1e-4，色度分母亮度下限12/255；启发式掩码不是材质/噪声真值。
信号maxRGB 4/255→12/255线性置信度，排除饱和邻域(250/255)、2像素边界及coverage<.95。
空掩码返回0，按有效权重归一。未使用额外逐像素色度目标、Laplacian或随机B0点对平滑。

R 范围/表达稳定：B0越出[log(.001),0]平方 1e-5，tanh(detail)² 1e-6，tanh(decoder)² 2e-5。
论文 LASG：所有显式lobes的平均幅度 1e-4、平均双轴锐度 5e-5、
平均 max(lambda_x,lambda_y)/(min(lambda_x,lambda_y)+1e-6) 1e-5。
LASG仅更新幅度/锐度，不直接更新方向、切向量和底光；不开启SG尾部或额外L纹理项。
新增项1000–2000步线性开启，独立白名单梯度：R仅B0/detail/decoder，L仅ASG幅度/锐度。
辅助R冻结几何/相机/共享特征，不影响屏幕梯度/densification；主监督保留原版权限。

原版 L1/DSSIM、光照、深度、StableSR双向detach、增强、residual、体积损失保持。
最终渲染默认关闭单residual。所有cfg/实际权重/每100步损失和梯度写入模型目录。
新实验两组fresh seed0、chair 32/5、8000步；主评估50步pose，固定/500步单列。
工程通过不保证物理解耦或视觉改善。论文均值LASG可能压低方向响应与各向异性，必须报告实测。
