# Game Model Learning for Mean Field Games

## 论文信息

- 标题: `Game Model Learning for Mean Field Games`
- 页数: 13 页
- 当前读取来源: `Game_Model_Learning_for_Mean_Field_Games__Copy_(1).pdf`
- 备注: PDF 中未提取到作者元信息，首页显示为 `Paper #86`，看起来像匿名评审版本

## 一句话总结

这篇论文想解决的问题是: 在 Mean Field Game (MFG) 里，精确计算“某个策略在某个 mean field 下的收益”很贵，于是作者提出先从仿真数据里学一个近似的 utility model，用它替代大量昂贵的真实求值，从而支持后续的均衡搜索和经验博弈分析。

## 中文精炼总结

### 1. 论文要解决什么问题

在有限博弈里，我们常常可以先收集一些策略组合的收益数据，再训练一个 game model，之后在这个近似模型上做均衡分析。但在 MFG 里，这件事更难，因为收益函数不仅依赖一个代表玩家的策略，还依赖整个无限人群诱导出来的 mean field，而且这个 mean field 通常还是随时间变化的。

困难主要有两个:

1. `strategy` 和 `mean field / distribution` 都是时序对象，直接当成回归输入会非常高维。
2. MFG 的 utility 对 mean field 一般不是线性的，所以不能像某些经验博弈那样简单做显式表格或线性组合。

论文的核心贡献就是把这个问题重新改写成一个“可学习”的回归问题。

### 2. 论文的核心思想

作者不再直接学习原始形式的

`u(s, mu)`

而是把它改写成一个黑盒模型

`u_hat(I(s), sigma)`

其中:

- `I(s)` 是 pure strategy 的离散编号，通常用 one-hot 表示
- `sigma` 是限制策略集 `Lambda` 上的 mixed strategy 概率向量
- `sigma` 通过前向分布演化会唯一决定诱导出的 mean field `mu_sigma`

这样做以后，原本复杂的时序策略和时序分布，不再显式展开成长向量，而是被“策略编号 + 混合权重”替代。这就是论文里的 **coarse coding**。

### 3. 这篇论文为什么有价值

如果这个 learned utility 足够准，那么:

- 内层均衡求解就可以直接在模型上算，不必反复精确诱导 distribution
- Fictitious Play、Replicator Dynamics 这类动态过程会更便宜
- EGTA 这类需要反复评估策略收益的流程会明显加速

论文的立场不是“直接学习 equilibrium”，而是“先学习 utility model，再把它交给博弈分析算法使用”。

## 方法拆解

### 1. 先定义 restricted game

作者并不是在完整无限策略空间里直接学，而是先用 iterative EGTA / Double Oracle 风格的方法，生成一个有限的策略池 `Lambda`。这个策略池应该:

- 足够多样
- 包含重要的战略交互
- 最好覆盖真实 MFG 均衡的 support

也就是说，论文先做“缩小策略空间”，再在这个小空间里学习 utility。

### 2. coarse coding 是什么

这是整篇论文最关键的技巧。

原始目标是学习:

`u_tilde(s, mu_sigma)`

但 `s` 和 `mu_sigma` 都带时间维度，直接输入神经网络会很难。于是作者改用:

- pure strategy 用索引 `I(s)` 表示
- mixed strategy 用概率向量 `sigma` 表示

于是学习目标变成:

`u_hat(I(s), sigma) ≈ u_tilde(s, mu_sigma)`

你可以把它理解成:

- “这个 pure strategy 是谁” -> 用 one-hot 告诉模型
- “当前总体人群是怎样混合这些策略的” -> 用概率向量告诉模型
- “在这种人群背景下，这个策略的 payoff 是多少” -> 由模型输出

### 3. 为什么 coarse coding 成立

作者依赖一个重要前提:

- 初始分布 `mu_0` 固定
- forward equation 是确定性的

因此，对给定的 mixed strategy `sigma`，诱导出的 `mu_sigma` 是唯一确定的。

所以虽然网络输入里没有显式放入完整的时间序列分布，`sigma` 仍然足以作为 `mu_sigma` 的代理表示。换句话说，模型是在隐式学习:

`sigma -> mu_sigma -> utility`

这也是论文所说的 end-to-end learning 味道最强的地方。

### 4. 训练数据怎么构造

一条训练样本长这样:

- 输入: `I(s), sigma`
- 标签: `u_tilde(s, mu_sigma)`

具体步骤是:

1. 在 restricted strategy set 上采样很多 mixed strategy `sigma`
2. 用 forward equation 诱导出对应 distribution `mu_sigma`
3. 对每个 pure strategy `s in Lambda`，计算真实 utility
4. 收集成监督学习数据

### 5. 为什么要特别设计采样

如果策略空间维度高，只做普通均匀采样会出现一个经典问题:

- 样本容易集中在 simplex 中间
- 角落区域覆盖不足

而均衡、极端策略组合、best response 往往恰恰跟这些边角区域有关。

所以作者组合了两种采样:

#### Grid sampling

在 simplex 表面铺规则网格点。优点是:

- 明确覆盖边界和角点附近
- 对低到中等维度空间比较稳定

论文中给了组合数公式，实验里使用 `K = 4` 的网格密度。

#### Symmetric Dirichlet sampling

从多个不同浓度参数 `alpha` 的对称 Dirichlet 分布采样。

- `alpha > 1`: 样本偏向 simplex 中心
- `alpha < 1`: 样本更靠近角点

作者让 `alpha` 在一个范围里变化，这样能同时覆盖中心区域和边缘区域。

#### 两者结合的意义

我对这部分的理解是:

- grid 负责“结构化覆盖”
- 多组 Dirichlet 负责“随机填充不同密度区域”

这比只用一种采样更适合高维 mixed strategy 空间。

### 6. 模型结构

论文使用的是比较直接的神经网络回归器:

- 输入: `one-hot(strategy index) + mixed strategy probabilities`
- 输出: 一个标量 utility

附录里的超参数是:

- 2 个全连接隐藏层
- 每层 256 单元
- 激活函数 `ReLU`
- 学习率 `0.001`
- batch size `32`
- training steps `1000`

### 7. 学到模型后怎么用

作者主要拿它支持两种均衡搜索动态:

- Fictitious Play (FP)
- Replicator Dynamics (RD)

在传统 MFG 里，FP/RD 的某些步骤很贵，因为要:

- 合并策略
- 从 merged policy 重新诱导 distribution

而用了 coarse coding 之后，混合策略本身就是一个概率向量，所以:

- 平均策略更新变成更新权重向量
- 不需要每一步都显式重新做完整分布诱导

这是论文强调的第二个价值: 不只是“预测 utility”，还是“简化求解流程”。

## 实验结果怎么理解

### 1. 测试环境

作者在三类 MFG 上做实验:

- Linear Quadratic MFG
- 1-D Crowd Modeling / Beach Bar
- 2-D Crowd Modeling / Beach Bar

这些任务从简单到复杂，策略空间规模也逐渐增大。

### 2. utility 回归效果

论文报告的 `R^2` 很高:

- Linear Quadratic: `0.99998 ± 0.00001`
- 1-D Crowd Modeling: `0.9724 ± 0.0024`
- 2-D Crowd Modeling: `0.9465 ± 0.0034`

结论是:

- 在相对规则的环境里，模型几乎能完美拟合
- 在更复杂的 crowd modeling 里，虽然误差变大，但仍然足以支持后续分析

### 3. 对均衡搜索有没有帮助

作者把 learned utility 和 true utility 分别带入 FP / RD，对比 regret curve。

主要结论:

- 在 FP 中，learned utility 产生的 regret 曲线很快和真实曲线重合，并收敛到 0
- 在 RD 中，线性二次模型表现也很好
- 在 1-D / 2-D crowd 中，RD 在后期会有轻微偏离，但在均衡附近补采样并 fine-tune 后，可以继续收敛

这说明:

- 模型在“路径上的预测”总体是靠谱的
- 但在均衡附近，高精度仍然重要
- 必要时可以做局部再采样修正

### 4. 对 mean field 的估计是否靠谱

作者还比较了 learned utility 和 true utility 导出的 equilibrium distribution。

结论是:

- 两者在图像上几乎看不出差别
- Wasserstein distance 都很小，文中说小于 `0.0006`

这说明虽然模型没有显式输出完整 distribution，它依然足够准确地支撑最终均衡分布的恢复。

## 这篇论文的主线逻辑

你可以把整篇论文压缩成下面这条链:

1. 用 EGTA / Double Oracle 得到有限策略池 `Lambda`
2. 在 `Lambda` 上采样很多 mixed strategy `sigma`
3. 对每个 `sigma` 做 forward equation 得到 `mu_sigma`
4. 计算每个 pure strategy 在 `mu_sigma` 下的真实 utility
5. 训练 `u_hat(I(s), sigma)`
6. 在后续 FP / RD / EGTA 分析中，用 `u_hat` 替代大量真实 utility 查询

## 和你当前代码目录的对应关系

这部分对你应该最实用。

### 1. 论文里的 coarse coding，对应你现在代码里的输入拼接

在 `se_gm/meta_strategies.py` 里，已经有很直接的实现痕迹:

- `one_hot_policy`
- `weights`
- `X = np.append(one_hot_policy, weights)`

这和论文的

`I(s) + sigma -> utility`

几乎是一一对应的。

### 2. 论文里的 utility regressor，对应当前项目里的神经网络

在 `se_gm/model.py` 里，模型是一个简单的 MLP:

- 多层全连接
- `ReLU`
- 最后一层输出标量

这和论文附录中的网络配置方向一致。

### 3. 论文里的“先跑 EGTA，再采样 utility”，对应当前数据生成流程

在 `model_learning/generate_data.py` 里，流程很清楚:

1. 先跑 `egta_solver.iteration()`
2. 再启动 `Coarse_Utility_Sampler`
3. 对 restricted game 做 coarse data 采样

这和论文第 4.3 节完全同路子。

### 4. 论文里的“模型辅助均衡搜索”，对应 `se_gm` 模块

从目录命名和代码内容看:

- `egta.py` 是外层 EGTA 主流程
- `distribution.py` 负责 distribution 演化
- `se_gm/meta_strategies.py` 里有利用模型预测 utility 的逻辑
- `model_learning/*` 负责训练数据生成和 utility 模型训练

也就是说，你这个仓库里已经把论文思路落成了一个很接近工程实现的版本。

## 论文的优点

### 1. 问题定义很准确

它不是泛泛地说“学习 MFG”，而是明确锁定到:

- restricted strategy set 上的 utility regression
- 让 learned model 真正服务博弈分析

这个切口很清晰，也很工程化。

### 2. coarse coding 非常实用

它不追求保留全部时序细节，而是用一个足够工作的表示把问题降维。这个设计特别适合工程实现，因为:

- 特征简单
- 网络简单
- 易于接入现有 EGTA 流程

### 3. 评价标准选得对

作者不只报告回归误差，还检查:

- regret 曲线是否一致
- 是否能支持 FP / RD 找均衡
- equilibrium distribution 是否接近真实结果

这些指标比单看 `R^2` 更能说明模型有没有“博弈意义”。

## 论文的局限

### 1. strategy index 不表达策略相似性

论文自己也承认这一点。策略只用编号表示的后果是:

- 模型知道“这是第 3 个策略”
- 但不知道“第 3 个和第 4 个策略其实很像”

所以模型很难泛化到新的未见策略。

### 2. 泛化范围主要还是 restricted game 内部

这篇论文学到的是:

- 在已有策略池里的策略
- 在这些策略张成的 mixed strategy 空间里的 utility

它不是在完整大空间里直接学一个通用 MFG 求解器。

### 3. 在均衡附近可能需要主动补样本

从 RD 的实验看，模型在关键区域仍可能需要:

- 局部再采样
- fine-tuning

这说明单次离线训练未必总够，在线修正可能是必要的。

## 对你读代码时的启发

如果你接下来要继续看这个 EGTA 目录，我建议重点带着下面几个问题去读:

1. 当前代码里的 `weights` 是否就是论文里的 `sigma`
2. 当前模型输入是否只使用 `one-hot + mixed weights`
3. 数据采样是否同时覆盖了 grid 和 Dirichlet
4. 模型是在 outer EGTA 的哪个阶段被训练和调用
5. 代码里是否实现了论文提到的“均衡附近再采样 / fine-tune”

如果这 5 个问题都对上了，你就基本把论文和实现串起来了。

## 我对这篇论文的整体理解

这篇论文最重要的不是提出了一个特别复杂的新网络，而是提出了一个很好的“接口重写”方法:

- 把难处理的时序策略和时序 mean field
- 压缩成策略编号和混合权重
- 再把 utility 学成一个可查询的黑盒

它真正解决的是 MFG 里 game model “怎么定义、怎么训练、怎么用于求解”的问题。对 EGTA 场景尤其合适，因为 EGTA 本来就天然需要:

- 一个有限策略池
- 大量收益查询
- 多轮均衡更新

所以这篇论文和你现在这套 `EGTA + model_learning + se_gm` 代码是非常贴近的。

## 如果继续深入，下一步最值得读什么

如果你还要继续深入，我建议按这个顺序看:

1. `model_learning/generate_data.py`
2. `model_learning/sample_utility.py`
3. `se_gm/model.py`
4. `se_gm/meta_strategies.py`
5. `egta.py`

这个顺序基本就是论文方法在代码里的执行链。
