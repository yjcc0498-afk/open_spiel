# V2 训练结果对比

## 实验背景

本次 V2 实验使用的是之前已经生成的 `300 samples` 数据集，还没有重新 generate 新版 `[30,11]` 策略特征。因此：

- `transformer_stats` 输入仍是旧版 `[30,6]` 策略行为特征
- 已使用 V2 的模型结构与训练策略，包括目标值标准化、特征标准化、AdamW、学习率调度、early stopping、best model 保存
- `transformer_stats` 使用 policy feature bank，将混合策略表示为纯策略嵌入的加权组合
- `one_hot` 仍保持论文式 baseline：`[纯策略 one-hot + 混合策略 σ] -> MLP -> utility`

## 核心结果

| 方法 | 划分方式 | Train MSE | Val MSE | Train R2 | Val R2 | Best Val MSE | Best Val R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer_stats | strategy_holdout | 10.8104 | 528.8581 | 0.9819 | 0.4720 | 466.5682 | 0.5342 |
| one_hot | strategy_holdout | 3.8332 | 946.1205 | 0.9934 | 0.0429 | 902.3364 | 0.0872 |
| one_hot | random split | 3.5750 | 15.9166 | 0.9951 | 0.9758 | 15.2684 | 0.9768 |

## 结果解读

在 `random split` 下，`one_hot` 表现非常强，Best Val R2 达到 `0.9768`。这是论文式评测下的正常结果，因为训练集和验证集都包含同一批 18 个纯策略，只是混合策略样本不同。此时 one-hot 策略编号本身就是非常强的身份特征，MLP 很容易记住每个纯策略的效用模式。

在更严格的 `strategy_holdout` 下，结果发生明显变化。`one_hot` 的 Best Val R2 只有 `0.0872`，final Val R2 为 `0.0429`，说明当验证集包含训练阶段没见过的纯策略时，one-hot 几乎不能泛化。原因很直接：one-hot 只知道“第几个策略”，没见过的策略编号无法通过行为相似性迁移。

`transformer_stats` 在同样的 `strategy_holdout` 设置下，Best Val R2 达到 `0.5342`，final Val R2 为 `0.4720`，明显高于 one-hot holdout。这说明 V2 的策略行为表示和混合策略嵌入确实提供了跨策略泛化能力。

## 关键结论

如果按论文常规 `random split` 评价，当前 one-hot 仍然是最强 baseline。

如果评价真正的泛化能力，即留出部分纯策略不参与训练，V2 transformer_stats 明显优于 one-hot：

```text
strategy_holdout Best Val R2:
transformer_stats = 0.5342
one_hot           = 0.0872
```

这说明 V2 没有在随机划分上超过论文 one-hot，但已经在“未见过纯策略泛化”这个更严格、更符合泛化目标的评测上超过 one-hot。

## 为什么 V2 有提升

V1 的 transformer 主要是把候选策略 `[30,6]` 编码成一个策略嵌入，再与混合策略向量拼接。V2 进一步把混合策略看作“纯策略嵌入的带权集合”，即：

```text
z_sigma = sum_j sigma_j * z_j
```

最终预测时使用：

```text
[z_i, z_sigma, z_i * z_sigma, |z_i - z_sigma|, sigma] -> MLP -> utility
```

这比单纯拼接 `sigma` 更符合 MFG/EGTA 的结构，因为效用本质上取决于“候选纯策略 i”和“群体混合行为 sigma”之间的关系。

## 当前局限

本次 V2 还没有使用新生成的 `[30,11]` 特征，只是在旧 `[30,6]` 数据上训练。因此当前结果主要验证了 V2 模型结构和训练流程有效，但还没有完全发挥新版特征提取器的能力。

此外，`transformer_stats` 在 holdout 下仍有过拟合迹象：Train R2 为 `0.9819`，Best Val R2 为 `0.5342`，说明模型已经学到了一部分跨策略规律，但验证集仍明显难于训练集。

## 后续建议

下一步应重新 generate 一版 `transformer_stats` 数据，使策略特征从 `[30,6]` 升级为 `[30,11]`。然后优先比较：

- `transformer_stats V2 + [30,11]` 在 `strategy_holdout` 下是否继续提升
- `transformer_stats V2 + [30,11]` 在 `random split` 下是否缩小与论文 one-hot 的差距
- 完整 `6000 samples` 下，V2 是否能进一步稳定泛化表现

当前最适合写入汇报的表述是：

```text
在论文式 random split 下，one-hot baseline 仍保持最优；
但在更严格的 strategy_holdout 泛化评测中，V2 transformer_stats 明显优于 one-hot，
说明基于策略行为序列的 Transformer 表示具备跨纯策略泛化能力。
```
