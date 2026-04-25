# EGTA 基础理解笔记

这份笔记的目标不是把所有数学细节讲完，而是帮助你先建立一个“这套代码到底在做什么”的整体认知。

---

## 1. 这段代码整体在做什么

这部分代码实现的是 **Mean Field Game (MFG)** 上的 **EGTA (Empirical Game-Theoretic Analysis，经验博弈分析)**。

可以把它理解成一个反复循环的过程：

1. 先有一小批候选策略。
2. 在这些候选策略组成的“经验博弈”里，找一个当前比较合理的混合策略。
3. 针对这个混合策略，再计算一个新的 best response（最佳响应）。
4. 把这个新策略加入候选策略池。
5. 重复以上步骤，让经验博弈越来越丰富。

所以它不是“一步直接求真均衡”，而是：

**边分析当前已有策略，边扩展策略集合。**

---

## 2. 什么是 Mean Field Game

Mean Field Game 可以粗略理解为：

- 有很多个体玩家；
- 单个玩家对整体影响很小；
- 但所有玩家的总体分布，会影响每个玩家的收益和状态转移。

因此，在 MFG 里一个策略通常不只是“怎么行动”这么简单，它还和：

- 当前所有玩家在状态空间中的分布；
- 分布随时间如何演化；
- 在这个分布下策略值是多少；

这些东西绑在一起。

所以你会在代码里经常同时看到：

- `policy`
- `distribution`
- `value`

这三者是配套出现的。

---

## 3. 什么是 EGTA

EGTA = 经验博弈分析。

它的核心思想是：

- 原始博弈太大，直接求解很难；
- 那就只拿一部分“已经发现的策略”出来，先形成一个小型经验博弈；
- 在这个小型博弈里求一个混合解；
- 再看有没有新的策略可以击败它；
- 如果有，就把新策略加入经验博弈继续分析。

所以 EGTA 的关键词是：

- **策略池扩张**
- **经验博弈**
- **元策略求解**
- **best response**

---

## 4. 什么是策略池

策略池就是当前已经收集到的一组策略。

在这份代码里，可以把它理解为：

- `self._policies`：当前已有的策略列表
- `self._distributions`：每个策略对应诱导出来的状态分布

一个策略加入策略池后，它以后就会参与“经验博弈”的混合求解。

---

## 5. 什么是经验博弈

经验博弈不是原始完整博弈，而是：

**只基于当前已知策略池构造出来的一个较小博弈。**

例如当前只找到 3 个策略：

- 策略 A
- 策略 B
- 策略 C

那么内层求解器就只在这 3 个策略上考虑混合，而不是在整个无限大的策略空间里搜索。

---

## 6. 什么是 meta-strategy（元策略）

meta-strategy 可以理解为：

**在策略池上的混合概率分布。**

也就是说，它回答的问题是：

- 当前经验博弈中，
- 应该给每个已有策略分配多少权重，
- 从而形成一个混合策略。

比如：

- 50% 选策略 A
- 30% 选策略 B
- 20% 选策略 C

这就是一个元策略。

在代码里，元策略求解器对应的是 `meta_strategies.py` 里的各种方法。

---

## 7. 什么是 best response

best response（最佳响应）指的是：

**在对手/总体分布固定时，自己能取得最高收益的策略。**

在 MFG 里，“对手”更准确地说是：

- 整体人群分布；
- 或者某个 merged policy 诱导出来的 mean-field distribution。

所以这套代码常做的事是：

1. 先得到一个混合策略；
2. 由它计算出一个分布；
3. 再对这个分布求 best response。

如果这个 best response 比当前混合策略更有利，就说明当前经验博弈还不够完整，需要把它加进去。

---

## 8. 什么是 distribution

distribution 就是：

**在某个策略下，玩家群体在各个状态上的概率分布。**

这是 MFG 和普通博弈很不一样的地方。

因为在 MFG 中，收益和状态转移往往依赖整体分布，所以仅有策略还不够，必须知道：

- 这个策略会把玩家群体带到哪些状态；
- 每个状态上大概有多少概率质量。

代码里 `DistributionPolicy` 做的就是：

**给定一个 policy，向前推进整局游戏，算出每个状态的分布。**

可以把它看成“前向传播”。

---

## 9. 什么是 value

value 就是：

**在给定策略和给定分布下，从某个状态开始能拿到的期望收益。**

代码里的 `PolicyValue` 做的是“价值评估”：

- 终止节点：直接取奖励；
- chance 节点：按概率取期望；
- mean-field 节点：先把 distribution 写进状态；
- decision 节点：按 policy 的动作概率继续递归。

可以把它看成“后向评估”或“递归期望计算”。

---

## 10. 什么是 merged policy

merged policy 指的是：

**把多个已有策略按某种权重混起来得到的策略。**

但这里不是简单线性平均，而是结合了：

- 当前 state 在不同 distribution 下出现的概率；
- 当前策略在这个 state 上的动作概率。

因此它更准确地表示：

**在某个状态条件下，综合多个候选策略后，最终应该采取什么动作概率。**

---

## 11. 什么是 inner loop

inner loop 就是：

**在当前策略池固定时，求当前经验博弈上的一个混合解。**

也就是说，inner loop 不负责发现新策略，它负责回答：

> 在“当前已有策略”里，最合理的混合方式是什么？

这个求出来的结果会被外层 EGTA 用来生成新的 best response。

---

## 12. 什么是 outer loop

outer loop 就是 EGTA 的主循环：

1. 跑 inner loop；
2. 得到当前经验博弈上的混合策略；
3. 对这个混合策略求 best response；
4. 把 best response 加进策略池；
5. 继续下一轮。

所以：

- **inner loop**：在已有策略里求混合
- **outer loop**：不断扩充策略池

---

## 13. 这份代码里默认用的 meta-strategy 方法

默认参数里写的是 `meta_strategy_method="nash"`，但这里要特别注意：

它并不是严格意义上的“精确 Nash 求解器”，而是映射到了：

- `FictitiousPlayMSS`

也就是说，这里的 `"nash"` 更像是：

**用 Fictitious Play 的方式，在经验博弈里近似找一个均衡混合策略。**

同时，这个文件里还提供了别的方法：

- `uniform`
- `RD`（Replicator Dynamics）
- `QRB`（Iterated Quantal Response）

---

## 14. Fictitious Play 在这里的直觉

Fictitious Play 的直觉是：

- 先有一个当前混合策略；
- 看看在这个混合策略下，哪个已有策略表现最好；
- 把“历史混合策略”和“这次找到的最好策略”再混一下；
- 重复进行。

时间越往后，历史平均的权重越大，新加进去的当前最好策略权重越小。

在这份代码里，它大致是在做：

- 评估每个已有策略的价值；
- 为每个 population 选一个当前最优策略；
- 再更新 merged policy。

---

## 15. consistency 是什么

代码里每次 outer iteration 后还会做一次 consistency evaluation。

你可以先把它理解成：

**用另一种更稳定的评估方式，再检查一下当前结果是否自洽。**

这里它默认用的是 `RD`（Replicator Dynamics）来做额外评估。

所以程序里会同时输出两类指标：

- 当前 EGTA 输出策略的 `NashConv`
- consistency policy 的 `NashConv`

---

## 16. NashConv 是什么

NashConv 可以理解为：

**距离均衡还有多远。**

更直觉地说：

- 如果任何玩家都很难通过单方面偏离获得更高收益，
- 那这个策略就更接近均衡，
- 对应的 NashConv 就更小。

因此通常：

- **越小越好**

---

## 17. 代码里的主模块分工

### `egta_example.py`

实验入口。

负责：

- 读取参数；
- 创建游戏；
- 创建 `MFGMetaTrainer`；
- 运行外层 EGTA 循环；
- 记录 NashConv 和日志。

### `egta.py`

外层控制器。

核心类是 `MFGMetaTrainer`，负责：

- 初始化策略池；
- 运行 inner loop；
- 调用 oracle 求 best response；
- 更新策略池。

### `inner_loop.py`

一个很薄的封装。

本身逻辑很少，主要就是调用 meta-strategy solver 的 `run()`。

### `meta_strategies.py`

内层求解器的核心文件。

负责：

- 在当前经验博弈上求混合策略；
- 支持 FP、RD、QRB、uniform 等方法；
- 输出当前 merged policy；
- 输出各原始策略的权重。

### `init_oracle.py`

初始化 best-response oracle。

默认是精确 BR：

- 先算 best-response value
- 再生成 greedy policy

### `distribution.py`

计算某个 policy 诱导出来的状态分布。

可以理解为：

- 给定策略；
- 沿着游戏树向前推进；
- 统计每个状态出现的概率。

### `policy_value.py`

计算策略价值。

可以理解为：

- 给定策略和分布；
- 沿着游戏树递归计算期望收益。

---

## 18. 你读代码时最容易卡住的地方

### 1）为什么 policy 和 distribution 总是一起出现

因为在 MFG 里：

- policy 会决定 distribution；
- distribution 又反过来影响收益和后续状态转移。

所以它们不是独立的。

### 2）为什么先 inner loop，再求 best response

因为 EGTA 不是直接在原始博弈里暴力找最优，而是：

- 先在当前已有策略组成的小博弈里找一个“临时均衡”；
- 再看这个临时均衡有没有漏洞；
- 用 best response 去暴露漏洞；
- 再扩张策略池。

### 3）为什么会有 merged policy

因为经验博弈中的解通常不是单个纯策略，而是多个策略的混合。

### 4）为什么会有很多“评估”

因为整套方法本质上是在近似：

- distribution 是近似构造的；
- mixed policy 是近似求的；
- equilibrium 也是经验博弈上的近似。

所以需要不断评估它是否足够好。

---

## 19. 一句话记住主流程

你可以把主流程记成：

**已有策略池 -> 内层混合求解 -> 得到 merged policy -> 计算其分布 -> 求 best response -> 把新策略加回策略池**

---

## 20. 建议你接下来怎么读代码

建议按这个顺序看：

1. `egta_example.py`
2. `egta.py`
3. `inner_loop.py`
4. `meta_strategies.py` 里的 `FictitiousPlayMSS`
5. `init_oracle.py`
6. `distribution.py`
7. `policy_value.py`

最关键的两个函数建议优先啃下来：

- `MFGMetaTrainer.iteration()`
- `FictitiousPlayMSS.iteration()`

这两个函数看懂以后，整个 EGTA 主线基本就通了。

---

## 21. 一个最简脑图

可以把它想成下面这个流程：

```text
已有策略池
   ↓
inner loop 求经验博弈混合解
   ↓
得到 merged policy
   ↓
计算 merged policy 对应的 distribution
   ↓
oracle 求 best response
   ↓
把新 best response 加入策略池
   ↓
继续下一轮 EGTA
```

---

## 22. 最后一句话

这部分代码最难的地方不是语法，而是概念耦合：

- 策略
- 分布
- 价值
- 元策略
- best response

它们在 MFG 里是一起运作的。

你先不要试图一次看懂所有细节，先牢牢记住下面这句最重要：

**EGTA 在这里做的是：不断在当前经验博弈里找混合解，再用 best response 扩充策略池。**
