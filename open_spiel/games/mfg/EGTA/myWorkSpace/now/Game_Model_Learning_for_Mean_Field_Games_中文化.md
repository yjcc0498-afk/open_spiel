# 均值场博弈的博弈模型学习

原文：`Game Model Learning for Mean Field Games`，Paper #86  
说明：本文档为 PDF 内容的中文化整理版。公式符号、算法名、引用编号和参考文献条目尽量保留原文形式；原 PDF 中的图像以图注形式标明。

## 摘要

本文提出一种从仿真数据中学习均值场博弈（Mean Field Games, MFGs）模型的方法。为了建模一个 MFG，核心问题是刻画一个代表性玩家在其他玩家无限总体分布，也就是均值场，给定时采用某一策略所产生的结果。因此，我们将问题定义为学习一个效用函数：它把受限策略集合与可能的均值场映射到效用值。

从 MFG 通常采用的时间相关结构出发，如果显式编码策略和均值场，回归问题会变得复杂到难以实施。为处理这一问题，我们提出一种粗编码（coarse coding）方案，它抽象掉时间相关复杂性，大幅简化输入表示。为了让学习到的效用函数能够在均值场空间中泛化，关键步骤之一是在受限策略空间中均匀采样能够诱导均值场的策略，而该空间通常是高维的。为在高维策略空间中进行采样，我们结合两类采样方案：网格采样，以及使用不同集中参数的 Dirichlet 分布采样。

通过在若干标准 MFG 上的实验，我们表明该方法能够实现有效泛化，且学习到的博弈模型可以成功支持博弈论分析。

## 1 引言

均值场博弈（MFGs）[12, 8] 描述了包含概念上无限多个相互作用的战略玩家的系统。总体中的玩家被视为原子化且同质的，因此系统状态可以由玩家状态上的概率分布，即“均值场”来刻画。MFG 分析可以归约为刻画单个代表性玩家在与整个总体交互时的最优行为，而总体由均值场表示。MFG 框架能够支持博弈论分析；若使用对应的大规模有限玩家博弈模型，这类分析通常不可处理。

尽管 MFG 提供了简化框架，MFG 中巨大的策略空间仍会阻碍博弈论分析。处理大策略空间的一种流行方法在有限博弈中已被广泛应用：构建一个近似真实博弈的博弈模型，并通过分析该博弈模型来研究真实博弈 [24, 23, 27]。博弈模型只涉及特定策略集合，以及由该策略集合组合上的样本效用诱导出的效用函数。由于限制了策略空间，博弈模型被视为真实博弈的简洁描述，也更容易分析。

然而，虽然这种方法在有限博弈中有效，却不能直接应用到 MFG，因为 MFG 中尚缺乏良好定义的博弈模型。定义博弈模型的关键步骤，是在受限策略集合上诱导一个效用函数。由于 MFG 的效用函数关于均值场通常是非线性的 [18, 26]，且均值场空间是连续的，因此若要显式表示任意策略混合所导出的均值场上的效用函数，完全不可行。没有合适的效用函数表示，已有工作中尚未为 MFG 给出良好定义的博弈模型，博弈分析方法因此无法直接应用。

本文填补这一空白，提出通过学习受限策略集合及其诱导分布上的效用函数，为 MFG 定义博弈模型。我们研究一般的 MFG 设置，其中策略和分布都随时间变化，即非平稳。若将它们显式编码为学习器输入，将导致维度高到不切实际。为处理时间相关性，我们提出一种编码方案，学习一个以策略和分布的充分统计式表示为输入并输出效用的函数。采用该方法后，时间相关策略和分布不再像真实效用函数那样被显式编码为输入，因此绕开了由时间相关性导致的表示复杂度。我们称该方法为粗编码。

为了学习有效的 MFG 效用函数，需要让学习到的效用函数具备跨越诱导均值场的策略空间的泛化能力。为此，训练数据集需要包含在受限策略空间中均匀采样的混合策略。对于大型 MFG，一个有效博弈模型通常包含几十个具有战略意义的策略，从而形成高维策略空间。为在这类高维空间中获得样本，我们提出结合两种采样方案：网格采样，以及使用不同集中参数的 Dirichlet 分布采样。结合粗编码与数据采样方法后，我们展示了该方法可以实现有效泛化，并准确预测效用。我们还表明，学习到的效用函数能够支持博弈论分析：虚拟博弈（Fictitious Play, FP）[21] 与复制子动力学（Replicator Dynamics, RD）[26] 在经验上都能借助学习到的效用函数收敛到纳什均衡。

本文的主要贡献是提出一种用于构建 MFG 博弈模型的学习方法。我们提出粗编码方案，成功处理了时间相关策略与分布的高维表示；提出一种结合网格采样与多参数 Dirichlet 分布采样的数据采样方案，以支持学习模型泛化；通过标准 MFG 实验证明，学习到的模型能够有效预测效用并具有良好泛化能力；最后展示该模型能够支持博弈论分析。

## 2 相关工作

### 2.1 有限博弈中的博弈模型学习

本质上，博弈模型学习是一种回归方法，用于在受限策略集合上学习效用函数。据我们所知，本文是第一项研究 MFG 中博弈模型学习的工作。下面简要回顾有限博弈中关于博弈模型学习的已有工作。

Vorobeychik 等 [25] 首次提出把标准形博弈模型学习视为效用函数回归，即从连续策略空间中采样得到的仿真数据中学习效用函数，并使用单参数策略表示展示了该方法。Ficici 等 [6] 根据包含策略剖面与效用的数据，将大量玩家聚类为两个角色，然后分别对每个角色应用效用函数回归。

Wiedenbeck 等 [28] 使用高斯过程回归（GPR）学习大型对称博弈的效用函数。回归器输入纯策略剖面并输出效用向量。在对称博弈中，效用函数依赖于有多少玩家选择每个策略，而不是哪些玩家选择，因此回归器的输入向量可以表示为一个非负计数向量，每个策略对应一个维度。学习到纯策略剖面的效用函数后，可以通过取期望扩展到混合策略剖面。作者还进一步研究了用神经网络进行回归。

Sokota 等 [22] 将博弈模型学习扩展到角色对称博弈，方法是回归偏离收益函数而不是收益函数。偏离收益可以直接被复制子动力学等算法使用，省去了先从收益函数推导偏离收益的步骤。Li 和 Wellman [15] 也采用该方法在贝叶斯博弈中学习纳什均衡。

也有一些工作关注在博弈的简洁描述上学习博弈模型。例如，Duong 等 [3] 和 Fearnley 等 [5] 从效用数据中学习图博弈模型 [9]。Li 和 Wellman [14] 将结构学习与收益回归结合起来，为多玩家博弈诱导出可处理的博弈模型。

对于 MFG，已有大量关于学习均衡的文献 [2, 7, 21, 17, 18, 13, 26]。虽然我们在实验中展示了学习到的效用函数可以促进均衡搜索，但本文方法在本质上不同于这些工作，因为我们的目标是学习 MFG 的效用函数，而不是提出一个均衡学习算法。因此，这些方法并不适合作为本文方法的公平基线。

### 2.2 MFG 中基于博弈模型的学习

Muller 等 [19] 将 Policy Space Response Oracles（PSRO）框架 [10] 适配到均值场博弈中，本质上是通过分析由强化学习构造策略集合的博弈模型来进行博弈论分析。然而，他们使用的博弈模型并没有被良好定义，因为 MFG 的效用函数一般不关于均值场线性，因此经验 MFG 模型无法被显式表示。

本文方法提供了一种通过学习构建博弈模型的实用方式。使用学习得到的博弈模型可能从两个方面显著降低经验博弈分析的成本。第一，学习到的博弈模型可以避免多次查询相同或相似的剖面。第二，学习到的博弈模型可以绕开分布诱导过程，从而加速经验博弈上的学习动态。

## 3 预备知识

我们研究具有时间结构的 MFG，其中代表性玩家在状态空间 `X`、动作空间 `A` 和时间跨度 `T` 上与总体交互。单总体均值场博弈的标准形表示为：

```text
G = (S, u)
```

总体对应概念上无限且可互换的一组玩家。`S` 表示策略集合，`u` 表示效用函数。状态空间 `X`、动作空间 `A` 和时间跨度 `T` 被编码在策略 `s ∈ S` 与效用函数 `u` 的定义中。

代表性玩家的策略将状态映射到动作空间中的动作分布。形式上，时刻 `t` 的策略 `s_t` 从状态空间 `X` 映射到动作分布空间 `Δ(A)`，其中 `Δ` 表示集合上的概率单纯形。整体策略 `s = (s_t)_{t∈[0,T-1]} ∈ S` 是从时刻 0 到时间跨度 `T` 的策略序列。

总体行为由均值场概括，均值场定义为博弈环境底层状态空间 `X` 上的分布。记时间跨度 `T` 上的总体分布为：

```text
μ = (μ_t)_{t∈[0,T]} ∈ Δ(X)^{T+1}
```

效用函数：

```text
u : S × Δ(X)^{T+1} → R
```

定义了总体中的代表性玩家在给定总体分布下采用某一策略时的收益。

混合策略 `σ` 是策略集合 `S` 上的概率分布，`σ(s)` 表示代表性玩家采用策略 `s` 的概率。在给定分布 `μ` 时，采用混合策略 `σ` 的期望效用为：

```text
u(σ, μ) = Σ_{s∈S} σ(s)u(s, μ).       (1)
```

代表性玩家对总体分布 `μ` 的最佳响应是任何使玩家收益最大化的策略，其中分布 `μ` 保持不变：

```text
br(μ) = argmax_{σ∈Δ(S)} u(σ, μ).
```

MFG 的纳什均衡（NE）是一个策略 `σ*`，满足 `σ* ∈ br(μ*)`，且 `μ*` 由 `σ*` 诱导。

称分布 `μ` 由 `σ` 诱导，记为 `μ^σ`，它遵循前向方程。给定初始分布 `μ_0`，对 `t ∈ [0, T-1]` 与所有 `x' ∈ X`：

```text
μ^σ_{t+1}(x') = Σ_{x,a∈X,A} μ^σ_t(x) σ_t(a | x) p(x' | x, a).       (2)
```

其中 `p : X × A → Δ_X` 是转移函数。令 `φ : Δ(S) × Δ(X) → Δ(X)^T` 表示一个函数，它以混合策略 `σ` 与初始分布 `μ_0` 为输入，输出诱导分布 `μ^σ`。

在博弈 `G` 中，代表性玩家在给定分布 `μ` 时采用 `σ` 的遗憾值定义为：

```text
ρ_G(σ, μ) = max_{s∈S} u(s, μ) - u(σ, μ).
```

遗憾值刻画了在给定分布 `μ` 时，代表性玩家若单方面从混合策略 `σ` 偏离到 `S` 中另一个策略，期望上最多可以获得多少收益。纳什均衡 `σ*` 对总体分布 `μ*` 的代表性玩家具有零遗憾。

博弈模型 `Ĝ_{S↓Λ} = (Λ, ũ)` 是真实博弈 `G` 的近似，其中 `ũ` 是 `u` 在受限策略空间 `Λ` 上的投影。

## 4 方法

### 4.1 时间相关的策略与分布

博弈模型学习的主要目标，是在受限策略集合上学习效用函数 `ũ(σ, μ)`。由于 `ũ(σ, μ)` 是 `ũ(s, μ)` 对 `s ∈ Λ` 的期望（由式 1 可得），因此只需在纯策略上学习效用函数 `ũ(s, μ)`。在 MFG 中，策略 `s` 与分布 `μ` 通常都随时间变化，即：

```text
s = (s_t)_{t∈[0,T-1]},    μ = (μ_t)_{t∈[0,T]}.
```

如果将它们显式编码为学习器输入，就会产生高维特征向量，使回归设置复杂到难以实际使用。为处理这一问题，我们提出粗编码方案。

### 4.2 粗编码

在有限博弈中，效用函数可以被视为黑盒（图 1），从而抽象掉策略细节以及效用计算机制。一个黑盒效用函数以策略剖面的简单表示 `I(s)` 为输入，例如每个玩家一个策略索引构成的向量，并输出效用样本。

图 1：黑盒效用函数。

受这种抽象启发，我们通过学习真实效用函数 `ũ` 的黑盒版本来处理时间相关性。数学上，考虑受限策略集合 `Λ ⊂ S`。令：

```text
I : Λ → Z+
```

为索引函数，它为每个策略 `s ∈ Λ` 分配一个唯一索引。令 `σ` 为诱导分布 `μ^σ` 的混合策略。由于分布诱导函数 `φ` 是确定性的，在固定初始分布 `μ_0 ∈ Δ(X)` 与策略 `σ` 后，分布 `μ^σ` 被唯一确定。因此，我们不学习带有时间相关输入的 `ũ(s, μ^σ)`，而是学习黑盒效用函数：

```text
û : I(Λ) × Δ(Λ) → R
```

其中 `I(s)` 与 `σ` 分别作为 `s` 与 `μ^σ` 的等价表示。我们称这种表示为粗编码。

我们的目标是用 `û(I(s), σ)` 预测真实效用 `ũ(s, μ^σ)`，因此最小化均方损失：

```text
E[(ũ(s, μ^σ) - û(I(s), σ))^2].
```

回归基于神经网络，其结构如图 2 所示。实践中，`I(s)` 可以是任意类别输入表示，例如 one-hot 编码；`σ` 是策略概率向量。

图 2：用于粗编码的神经网络结构。

粗编码有两个主要优点。第一，它只需要简单的网络结构：一个 one-hot 编码策略和一个混合策略的向量表示。因此粗编码实现起来很直接。

第二，粗编码可以大幅简化 MFG 中均衡搜索算法的执行。以 FP 为例，MFG 的 FP 可以概括为三个迭代步骤：计算给定分布 `μ` 下的最佳响应策略；加入新的最佳响应策略后更新平均策略；通过前向方程（式 2）计算平均策略诱导的分布 `μ`。将多个策略平均为一个合并策略以及分布诱导，即第二步和第三步，通常计算成本较高 [13]。

使用粗编码时，第二步和第三步的计算可以大幅减少。第二步中，平均策略现在由 `Λ` 上的概率向量表示，而不是由一个策略实体表示；平均策略更新也就变成了概率向量更新。因此不再需要跨状态、动作和时间跨度合并策略。与此同时，在第三步中，由于概率向量，也就是混合策略 `σ`，已经能够充分表示分布 `μ^σ`（初始分布 `μ_0` 固定且前向方程确定），因此不再需要显式进行分布诱导。本质上，基于粗编码的学习可以被视为端到端学习，其中分布诱导被隐式地从数据中学到。基于这些优点，给定粗编码效用模型后，FP 可以被显著加速。

### 4.3 数据采样

在本文的回归任务中，一个数据点由纯策略索引 `I(s)`、混合策略 `σ` 和真实效用 `ũ(s, μ^σ)` 构成。为了收集这些数据点，基本要求是采样得到的混合策略 `σ` 应当在受限策略空间中均匀分布，从而使学习器能够在诱导分布空间中泛化。对于大型 MFG，博弈模型通常包含几十个策略，使样本空间成为高维空间。众所周知，高维空间中的均匀采样会遭遇维度灾难，有限数量的样本主要集中在空间中心，而角落区域样本较少。为处理这一问题，我们结合两种采样方案。

第一种采样方案是网格采样。在网格采样中，会在策略单纯形表面采样一组网格点。网格采样可以通过组合算法 [20] 实现。具体地，令参数 `K` 为每个未归一化样本的元素和，`|S|` 为受限集合中的策略数量。样本总数为：

```text
(K + |S| - 1)! / (K!(|S| - 1)!)
```

也就是组合数 `C(K + |S| - 1, |S| - 1)`。例如，当 `K = 4` 且 `|S| = 2` 时，样本向量为：

```text
[(0,4), (1,3), (2,2), (3,1), (4,0)].
```

每个样本除以 `K` 归一化后，在 `|S|`-单纯形上生成网格：

```text
[(0,1), (1/4,3/4), (2/4,2/4), (3/4,1/4), (1,0)].
```

在实验中，我们选择 `K = 4`，`|S|` 根据具体 MFG 而变化。

第二种采样方案依赖具有不同集中参数 `α` 的对称 Dirichlet 分布。数学上，对称 Dirichlet 分布的密度函数可用 Gamma 函数表示为：

```text
f(x_1, ..., x_|S|; α) = Γ(α|S|) / Γ(α)^|S| × Π_i x_i^{α-1}.
```

集中参数 `α` 控制样本密度。当 `α > 1` 时，样本在质心附近更密集；当 `α < 1` 时，样本分布更稀疏，并更接近单纯形角落。我们设置一系列 `α` 值，并从相应 Dirichlet 分布中采样，目标是生成足够多能够覆盖策略单纯形的样本。

结合这两种采样方案得到的样本，我们获得了策略空间中的一组混合策略样本。随后，对于每个采样得到的 `σ`，通过前向方程诱导 `μ^σ`，并对每个纯策略 `s ∈ Λ` 与 `μ^σ` 评估 `ũ(s, μ^σ)`。因此，我们得到包含纯策略索引 `I(s)`、混合策略 `σ` 以及对应效用 `ũ(s, μ^σ)` 的数据点。

以上讨论假设已经存在一个受限策略集合，而我们的目标是在该集合上学习效用函数。为了获得这样的策略集合，我们将迭代式 EGTA 应用于 MFG，即 MFG 上 Double Oracle [16] 的一种变体 [18, 26]。这是一种常见方法，用于为博弈模型组装策略组合。生成的策略既具有多样性，也表现出有趣的战略交互，例如包含真实 MFG 的纳什均衡。

### 4.4 近似纳什均衡

学习到的效用函数 `û` 是否足够准确以支持 NE 计算，对分析 MFG 至关重要。为进行评估，我们考虑两种学习动态：FP 与 RD，并使用学习到的效用函数实现它们。对于每种学习动态，我们分别使用真实效用函数和学习到的效用函数测量该动态生成的中间策略的遗憾值。如果两者给出的遗憾值彼此接近，并且都收敛到 0，也就是接近 NE，则可以认为学习到的效用函数预测良好，并且能够支持 NE 计算。

算法 1 与算法 2 给出了使用学习效用函数实现 FP 与 RD 的方式。与使用真实效用函数实现 FP 与 RD 相比（Wang and Wellman [26] 的算法 2 和算法 3），由于粗编码的存在，使用学习效用函数的实现显著简化。

#### 算法 1：使用学习效用函数的虚拟博弈

输入：学习到的效用函数 `û`。将初始策略 `σ̄` 定义为受限集合 `Λ = (s_1, ..., s_n)` 中策略的平均。

```text
for iteration j ∈ {1, ..., J}:
    评估 û(I(s), σ̄), 对所有 s ∈ Λ
    选择最佳响应策略 s = argmax_{s'∈Λ} û(I(s'), σ̄)
    更新 σ̄: σ̄(s) = (1 / j) n_s, 对所有 s ∈ Λ
        其中 n_s 是策略 s 被选为最佳响应的次数
return σ̄
```

#### 算法 2：使用学习效用函数的复制子动力学

输入：学习到的效用函数 `û`。将初始策略 `σ̄` 定义为受限集合 `Λ = (s_1, ..., s_n)` 中策略的平均；学习率 `dt`。

```text
for iteration j ∈ {1, ..., J}:
    评估 û(I(s), σ̄), 对所有 s ∈ Λ
    计算平均适应度 F = Σ_{s∈Λ} σ̄(s) û(I(s), σ̄)
    for s ∈ Λ:
        更新 σ̄'(s) = σ̄(s) + dt · σ̄(s)[û(I(s), σ̄) - F]
    σ̄ = σ̄'
return σ̄
```

## 5 实验结果

### 5.1 实验环境

#### 5.1.1 沙滩酒吧问题

我们考虑沙滩酒吧问题 [1, 4]，并采用 Perrin 等 [21] 的模型。MFG 的沙滩酒吧问题是一个具有状态空间 `X` 的马尔可夫决策过程。状态空间可以定义为一维或二维，从而得到两类复杂度不同的沙滩酒吧问题。

在一维沙滩酒吧问题中，状态定义在一维环面上：

```text
X = {0, ..., |X|-1}
```

如图 3 所示。玩家动作包括保持不动 `a_t = 0`，向左移动 `a_t = -1`，或向右移动 `a_t = 1`。玩家在环面上移动，目标是在尽可能靠近酒吧的同时避开拥挤区域。

在二维沙滩酒吧问题中，状态定义在二维环面上，玩家拥有额外动作，例如向上、向下移动。动作数量增加会导致更多策略，使二维沙滩酒吧问题在战略上比一维问题更复杂。

在两个问题中，MFG 代表性玩家的转移函数为：

```text
x_{t+1} = x_t + a_t + ξ_t
```

其中 `a_t` 是代表性玩家在时刻 `t` 的动作，`ξ_t` 表示环境随机性。即时奖励函数为：

```text
r(x_t, a_t, μ_t) = r̃(x_t) - |a_t| / |X| - log(μ_t(x_t)).
```

其中 `r̃(x_t)` 衡量状态 `x_t` 到酒吧的接近程度；`-|a_t|/|X|` 是运行成本，若采取任何动作则为负；`-log(μ_t(x_t))` 表示玩家对拥挤区域的厌恶。

图 3：沙滩酒吧问题示意图。

#### 5.1.2 线性二次 MFG

线性二次模型是 MFG 研究社区最早得到完整数学处理的模型之一。本文采用 Perrin 等 [21] 的模型。具体而言，线性二次 MFG 定义在离散状态空间：

```text
X = {-L, ..., L}
```

动作空间为 `{-M, ..., M}`，代表性玩家可以向左或向右最多移动 `M` 个状态，也可以保持不动。转移函数为：

```text
x_{t+1} = x_t + (K(m_t - x_t) + a_t)δt + zξ_tδt
```

奖励函数为：

```text
r(x_t, a_t, μ_t) = [-1/2 |a_t|^2 + q a_t(m_t - x_t) - κ/2 (m_t - x_t)^2]δt
```

终端奖励为：

```text
r(x_T, a_T, μ_T) = -C/2 (m_T - x_T)^2.
```

其中 `K, q, κ, C` 为给定的非负常数，`ξ_t` 表示由常数 `z` 正则化的环境随机性。`δt` 衡量两个时间步之间的时间间隔，`m_t = Σ_{x∈X} x μ_t(x)` 是状态期望。该奖励函数鼓励玩家跟随总体的平均状态。

### 5.2 训练效用函数

训练数据集包含两部分：采样精度 `K = 4` 的网格样本，以及 `α ∈ (0,1]`、步长为 0.05 的 Dirichlet 样本。为了构造训练数据点，对于每个 `α`，我们从对应 Dirichlet 分布中采样 150 个混合策略 `σ`，并对每个 `s ∈ Λ` 计算 `ũ(s, μ^σ)`。测试数据从范围更大的 Dirichlet 分布中采样，其中 `α ∈ (0.01, 2.01]`，步长为 0.1。对于每个 `α`，同样采样 150 个混合策略来构造测试数据点。训练细节见附录 A。

表 1 报告了学习到的效用函数在上述三个 MFG 的测试数据集上的 `R²` 分数。`R²` 分数衡量模型解释的总变差，数学上为：

```text
R² = 1 - Σ_k(ŷ_k - y_k)^2 / Σ_k(y_k - ȳ)^2.
```

其中 `k` 是样本索引，`ŷ` 是模型估计值，`ȳ` 是目标 `y` 的均值。`R²` 分数介于 0 与 1 之间。若 `R²` 接近 1，说明数据变差被模型很好地解释。由表 1 可见，本文模型在三个 MFG 中都取得了很高的 `R²` 分数。

表 1：测试结果。

| MFG | R² 分数 |
|---|---:|
| Linear Quadratic | 0.99998 ± 0.00001 |
| 1-D Crowd Modeling | 0.9724 ± 0.0024 |
| 2-D Crowd Modeling | 0.9465 ± 0.0034 |

### 5.3 使用学习效用函数近似 NE

#### 5.3.1 使用学习效用函数的 FP

图 4 分别绘制了在三个 MFG 中使用真实效用函数与学习效用函数运行 FP 得到的遗憾曲线。由于这些 MFG 支持精确策略评估，例如通过动态规划，遗憾曲线可以被精确计算，因此图中没有报告误差条。在所有情况下，我们都观察到由学习效用函数生成的遗憾曲线能够快速与真实效用函数生成的曲线重合，并且二者均成功收敛到 0，即达到 NE。这说明学习到的效用函数在 FP 的均衡搜索路径上具有良好预测精度，并能够支持 MFG 的博弈论分析。

图 4：使用 FP 得到的遗憾曲线。

#### 5.3.2 使用学习效用函数的 RD

图 5 绘制了三个 MFG 中 RD 的遗憾曲线。在线性二次博弈中，我们再次观察到两条遗憾曲线快速重合，RD 成功收敛到 0。然而，在一维与二维拥挤建模博弈中，使用学习效用函数的遗憾曲线起初像通常一样快速下降，随后轻微发散。这里的“轻微发散”表示由发散造成的误差仍远小于效用尺度，因此可能不会显著影响进一步的博弈论分析。

这种发散源自学习效用函数在均衡附近的预测误差。为了提高学习效用函数在均衡附近的准确性，我们在发散点附近的 `σ` 邻域中重新采样效用 `ũ(s, μ^σ)`，并微调学习到的效用函数。随后，使用微调后的学习效用函数继续 RD。在图中，我们在第 19 次迭代重新采样（红色竖线标示），并观察到发散很快消失，RD 收敛到 NE。这再次表明，学习到的模型能够支持 MFG 的博弈论分析。

图 5：使用 RD 得到的遗憾曲线。

### 5.4 均值场估计

为了验证学习到的博弈模型能够准确估计均值场，即分布，我们分别绘制了使用真实效用函数和学习效用函数计算出的均衡策略所诱导的时间相关分布。图 6 与图 7 展示了一维和二维拥挤建模博弈中，按步长 3 选取的均衡分布。为便于可视化，一维博弈中的状态被重塑为二维形式。

通过分别比较图 6 与图 7 中上方和下方的图像，可以观察到，使用学习效用函数生成的分布与使用真实效用函数生成的分布在视觉上几乎无法区分。这种准确性可通过 Wasserstein 距离量化。表 2 与表 3 报告的距离都很小（小于 0.0006），尽管随着时间推移存在增大的趋势。

图 6：一维拥挤建模中的分布估计。上两行为真实效用函数；下两行为学习效用函数。  
图 7：二维拥挤建模中的分布估计。上两行为真实效用函数；下两行为学习效用函数。

表 2：一维拥挤建模博弈中的 Wasserstein 距离（×10^-4）。

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Values | 0.0 | 4.0 | 7.0 | 9.0 | 1.1 | 1.6 | 1.8 | 1.9 | 2.2 | 2.0 |

| t | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Values | 1.8 | 1.8 | 1.7 | 2.2 | 2.3 | 2.1 | 2.0 | 2.0 | 1.9 | 2.2 |

| t | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Values | 2.3 | 2.6 | 3.2 | 3.1 | 3.5 | 3.8 | 4.0 | 4.1 | 4.1 | 4.1 |

表 3：二维拥挤建模博弈中的 Wasserstein 距离（×10^-4）。

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Values | 0.0 | 2.4 | 3.6 | 3.5 | 3.3 | 3.1 | 3.7 | 3.6 | 3.6 | 3.7 |

| t | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Values | 3.9 | 3.9 | 3.8 | 3.0 | 3.5 | 3.9 | 4.1 | 3.6 | 3.9 | 5.3 |

| t | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Values | 4.9 | 5.1 | 4.9 | 4.1 | 3.9 | 4.0 | 4.0 | 4.3 | 4.7 | 4.6 |

## 6 结论

本文为 MFG 提出了一种博弈模型学习方法。我们引入粗编码方案，用于处理 MFG 效用函数中的高维输入；同时提出一种面向高维策略空间的数据采样方案。实验显示，使用学习到的效用函数生成的学习曲线几乎与真实效用函数给出的曲线重合，这表明学习效用函数准确，并能够支持 MFG 中的博弈论分析。通过结合总体、受限策略集合和学习到的效用函数，MFG 的博弈模型得以被良好定义。

在本文的博弈模型学习方法中，每个策略由一个索引表示。虽然索引编码有利于处理时间相关性，但它并不提供策略之间相似性的信息。未来研究将包括用参数化策略的嵌入表示，即摘要表示，替代索引表示，从而使学习到的博弈模型能够泛化到新策略。

## 参考文献

[1] W. B. Arthur. Inductive reasoning and bounded rationality. American Economic Review, 84(2):406-411, 1994.

[2] P. Cardaliaguet and S. Hadikhanloo. Learning in mean field games: the fictitious play. ESAIM: Control, Optimisation and Calculus of Variations, 23(2):569-591, 2017.

[3] Q. Duong, Y. Vorobeychik, S. Singh, and M. P. Wellman. Learning graphical game models. In Twenty-First International Joint Conference on Artificial Intelligence, pages 116-121, Pasadena, 2009.

[4] J. Farago, A. Greenwald, and K. Hall. Fair and efficient solutions to the santa fe bar problem. In Grace Hopper Celebration of Women in Computing. Citeseer, 2002.

[5] J. Fearnley, M. Gairing, P. W. Goldberg, and R. Savani. Learning equilibria of games via payoff queries. J. Mach. Learn. Res., 16:1305-1344, 2015.

[6] S. G. Ficici, D. C. Parkes, and A. Pfeffer. Learning and solving many-player games through a cluster-based representation. In Twenty-Fourth Conference on Uncertainty in Artificial Intelligence, pages 187-195, 2008.

[7] X. Guo, A. Hu, R. Xu, and J. Zhang. Learning mean-field games. 33rd Annual Conference on Neural Information Processing Systems, 32, 2019.

[8] M. Huang, R. P. Malhame, P. E. Caines, et al. Large population stochastic dynamic games: Closed-loop McKean-Vlasov systems and the Nash certainty equivalence principle. Communications in Information & Systems, 6(3):221-252, 2006.

[9] M. Kearns. Graphical games. Algorithmic Game Theory, 3:159-180, 2007.

[10] M. Lanctot, V. Zambaldi, A. Gruslys, A. Lazaridou, K. Tuyls, J. Perolat, D. Silver, and T. Graepel. A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning. NIPS, 2017.

[11] M. Lanctot, E. Lockhart, J.-B. Lespiau, V. Zambaldi, S. Upadhyay, J. Perolat, S. Srinivasan, F. Timbers, K. Tuyls, S. Omidshafiei, et al. OpenSpiel: A framework for reinforcement learning in games. arXiv preprint arXiv:1908.09453, 2019.

[12] J.-M. Lasry and P.-L. Lions. Mean field games. Japanese Journal of Mathematics, 2(1):229-260, 2007.

[13] M. Lauriere, S. Perrin, S. Girgin, P. Muller, A. Jain, T. Cabannes, G. Piliouras, J. Perolat, R. Elie, O. Pietquin, et al. Scalable deep reinforcement learning algorithms for mean field games. arXiv preprint arXiv:2203.11973, 2022.

[14] Z. Li and M. Wellman. Structure learning for approximate solution of many-player games. In 34th AAAI Conference on Artificial Intelligence, volume 34, pages 2119-2127, 2020.

[15] Z. Li and M. P. Wellman. Evolution strategies for approximate solution of bayesian games. In 35th AAAI Conference on Artificial Intelligence, volume 35, pages 5531-5540, 2021.

[16] H. B. McMahan, G. J. Gordon, and A. Blum. Planning in the presence of cost functions controlled by an adversary. In 20th International Conference on Machine Learning, pages 536-543, 2003.

[17] R. K. Mishra, D. Vasal, and S. Vishwanath. Model-free reinforcement learning for non-stationary mean field games. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 1032-1037. IEEE, 2020.

[18] P. Muller, M. Rowland, R. Elie, G. Piliouras, J. Perolat, M. Lauriere, R. Marinier, O. Pietquin, and K. Tuyls. Learning equilibria in mean-field games: Introducing mean-field psro. In 21st International Conference on Autonomous Agents and Multi-Agent Systems, 2021.

[19] P. Muller, M. Rowland, R. Elie, G. Piliouras, J. Perolat, M. Lauriere, R. Marinier, O. Pietquin, and K. Tuyls. Learning Equilibria in Mean-Field Games: Introducing Mean-Field PSRO. In AAMAS, 2022.

[20] A. Nijenhuis. HS Wilf Combinatorial Algorithms. Academic Press, New York, 1975.

[21] S. Perrin, J. Perolat, M. Lauriere, M. Geist, R. Elie, and O. Pietquin. Fictitious play for mean field games: Continuous time analysis and applications. 34th Annual Conference on Neural Information Processing Systems, 2020.

[22] S. Sokota, C. Ho, and B. Wiedenbeck. Learning deviation payoffs in simulation-based games. In 33rd AAAI Conference on Artificial Intelligence, volume 33, pages 2173-2180, 2019.

[23] K. Tuyls, J. Perolat, M. Lanctot, E. Hughes, R. Everett, J. Z. Leibo, C. Szepesvari, and T. Graepel. Bounds and dynamics for empirical game theoretic analysis. In 19th International Conference on Autonomous Agents and Multi-Agent Systems, volume 34, page 7, 2020.

[24] O. Vinyals, T. Ewalds, S. Bartunov, P. Georgiev, A. S. Vezhnevets, M. Yeo, A. Makhzani, H. Kuttler, J. Agapiou, J. Schrittwieser, et al. StarCraft II: A new challenge for reinforcement learning. Technical Report 1708.04782v1, arXiv, 2017.

[25] Y. Vorobeychik, M. P. Wellman, and S. Singh. Learning payoff functions in infinite games. Machine Learning, 67:145-168, 2007.

[26] Y. Wang and M. P. Wellman. Empirical game-theoretic analysis in mean field games. arXiv preprint arXiv:2112.00900, 2021.

[27] M. P. Wellman. Methods for empirical game-theoretic analysis (extended abstract). In Twenty-First National Conference on Artificial Intelligence, pages 1552-1555, Boston, 2006.

[28] B. Wiedenbeck, F. Yang, and M. Wellman. A regression approach for modeling games with many symmetric players. In 32nd AAAI Conference on Artificial Intelligence, volume 32, 2018.

## 附录 A 实验细节

### A.1 参数

我们使用 OpenSpiel [11] 中的 MFG 环境。MFG 设置列于表 4。为了从具有大策略空间的完整 MFG 中构建有意义的受限博弈，我们将迭代式 EGTA 应用于 MFG [26]，并构建一个包含完整 MFG 的 NE 的受限博弈。换言之，NE 的支撑包含在策略集合中。

对于网格采样和 Dirichlet 分布采样，每一种方法都为 `ũ(s, μ^σ)` 中的每个 `s ∈ Λ` 贡献约 3000 个样本。本文选择的样本数量是充足的。未来研究中，一个很有意义的问题是考察样本数量与模型有效性之间的关系。

表 4：MFG 超参数。

| Games | # of States | Time Horizon | # of Strategies | # of samples |
|---|---:|---:|---:|---:|
| Linear Quadratic | 10 | 10 | 10 | ≈3000 |
| 1-D Crowd | 100 | 30 | 18 | ≈6000 |
| 2-D Crowd | 100 | 30 | 18 | ≈6000 |

神经网络参数列于表 5。神经网络的输入大小取决于 MFG。例如，一维拥挤建模博弈的输入大小为 36，也就是 one-hot 编码的策略数 18，加上诱导分布的混合策略支撑大小 18。输出大小为 1，即估计效用。

表 5：神经网络超参数。

| Parameters | Values |
|---|---|
| hidden layers | 2 fully connected layers with size 256 |
| activations | relu per hidden layer |
| learning rate | 0.001 |
| batch size | 32 |
| training steps | 1000 |
| validation set proportion | 0.3 |

### A.2 计算资源

实验使用一台 2 × 3.0 GHz Intel Xeon Gold 6154 CPU，内存 8 GB。

### A.3 代码可用性

根据双盲政策，作者将在评审后发布代码。

## 附录 B 均衡均值场估计

图 8：一维拥挤建模中使用真实效用函数的分布估计。  
图 9：一维拥挤建模中使用学习效用函数的分布估计。  
图 10：二维拥挤建模中使用真实效用函数的分布估计。  
图 11：二维拥挤建模中使用学习效用函数的分布估计。
