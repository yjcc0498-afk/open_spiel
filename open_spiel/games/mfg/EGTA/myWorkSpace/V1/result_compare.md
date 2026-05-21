# V1 训练结果对比

## 实验设置

- 游戏：1D Crowd，`size=100`，`horizon=30`
- 策略数：18 个纯策略
- 采样规模：`300 samples`，其中 grid 与 Dirichlet 各约 150
- 数据规模：`18 * 300 = 5400` 条 utility 样本
- 划分方式：训练集 4320，验证集 1080
- 对比方法：
  - `one_hot`：论文式输入 `[纯策略 one-hot + 混合策略 σ]`，接 MLP 预测效用
  - `transformer_stats`：`[30,6]` 策略行为特征经 Transformer 聚合为策略嵌入，再与混合策略 σ 拼接后接 MLP 预测效用

## 核心结果

| 方法 | Train MSE | Val MSE | Train R2 | Val R2 |
|---|---:|---:|---:|---:|
| one_hot | 72.9430 | 31.7521 | 0.8998 | 0.9518 |
| transformer_stats | 644.4898 | 847.9076 | 0.1346 | -0.2424 |

## 结论

在当前 V1 的 `300 samples` 小数据设置下，`one_hot` 明显优于 `transformer_stats`。`one_hot` 的验证集 R2 达到 `0.9518`，说明模型已经较好拟合了效用函数；而 `transformer_stats` 的验证集 R2 为 `-0.2424`，表示其预测效果低于直接预测均值的基线。

从误差看，`one_hot` 的验证 MSE 为 `31.7521`，远低于 `transformer_stats` 的 `847.9076`。这说明当前 Transformer 策略特征表示还没有充分保留纯策略身份与效用差异信息。

## 原因分析

`one_hot` 在小样本下更占优势，因为它直接给出“第几个纯策略”的身份信息。对于固定 18 个策略的 restricted game，策略编号本身就是强特征，MLP 很容易学习每个纯策略在不同混合策略下的效用模式。

`transformer_stats` 则把每个策略压缩成 `[30,6]` 的行为统计序列，再由 Transformer 聚合为 128 维嵌入。这个表示更有泛化潜力，但在 300 samples 下容易出现两个问题：一是统计特征可能不足以区分相近策略；二是 Transformer 参数量更大，小数据下训练不稳定。

## 当前判断

V1 结果不能说明 Transformer 思路无效，但可以说明：在当前特征设计和 300 samples 小数据下，`transformer_stats` 还不能替代论文中的 `one_hot` 表示。论文式 `one_hot + MLP` 是当前更强、更稳定的 baseline。

后续优化应优先考虑：

- 对 `utility_Y` 做标准化，提升训练稳定性
- 对 `[30,6]` 每个特征维度做标准化，避免尺度差异主导训练
- 增强策略行为特征，例如显式加入 `left_prob / stay_prob / right_prob`
- 在完整 `6000 samples` 数据上重新比较，观察 Transformer 表示是否随数据量增大而改善
