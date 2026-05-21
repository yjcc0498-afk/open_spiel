# D2 游戏说明（`mfg_crowd_modelling_2d`）

这份文档专门解释你项目里常说的 **D2 游戏**。  
这里的 D2，指的就是：

- `mfg_crowd_modelling_2d`

它是一个 **二维 Mean Field Game（MFG）人群移动游戏**，也常被叫做：

- **2-D crowd modelling**
- **2-D beach bar process**

---

## 1. 一句话先理解 D2

你可以把 D2 想成：

> 一大群人同时在一个二维网格地图上移动，大家都想靠近“有吸引力的位置”，但又不想和别人太拥挤，同时移动本身还有成本。

所以每个玩家都在做一个平衡：

- 想去“好地方”
- 不想挤
- 不想乱跑

---

## 2. D2 在你项目里的位置

### 环境定义

- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\games\mfg\crowd_modelling_2d.h`
- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\games\mfg\crowd_modelling_2d.cc`

### Python 侧辅助配置

- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\python\mfg\games\crowd_modelling_2d.py`

### EGTA 入口里怎么调用

- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\games\mfg\EGTA\egta_example.py`

你在 EGTA 里看到：

```bash
--game_name=mfg_crowd_modelling_2d
```

就是在跑这个 D2 游戏。

---

## 3. D2 的地图是什么样

这个游戏的地图是一个 **二维格子**。

默认参数通常是：

- `size=10`
- `horizon=10` 或 `20` 或 `30`

其中：

- `size=10` 表示地图是 `10 x 10`
- `horizon=30` 表示游戏一共进行 30 个时间步

另外这个地图是 **torus（环面）**：

- 从右边走出去，会从左边回来
- 从上边走出去，会从下边回来

所以它不是一个有真实边界的房间，更像一个“首尾相接”的平面。

---

## 4. D2 的动作有哪些

每一步玩家有 5 个动作：

- `down`
- `left`
- `stay` / `neutral`
- `right`
- `up`

也就是：

- 下
- 左
- 不动
- 右
- 上

所以这就是一个典型的二维移动控制问题。

---

## 5. D2 的游戏流程

这个游戏每一步大致是这样走的：

1. 玩家先选一个动作
2. 进入 mean-field 节点
3. 系统把当前整体人群分布 `distribution` 写入状态
4. 再经过 chance/noise 节点
5. 进入下一时刻

所以它不是普通单智能体网格世界，而是：

- 玩家决策
- 系统同步整体人群分布
- 再继续下一步

这正是 Mean Field Game 的核心味道。

---

## 6. D2 最重要的概念：distribution

在 D2 里，最关键的不是“某一个人在哪”，而是：

> 在整个地图上，每个格子里大概有多少玩家。

这个东西就是 **distribution（分布）**。

例如：

- 某个时间点，中心区域很多人
- 四周区域很少人

那么这个“地图热度图”就是 mean-field distribution。

在 MFG 里，玩家收益依赖这个分布，所以：

- 策略会影响分布
- 分布又会反过来影响策略价值

---

## 7. D2 的奖励由什么组成

这部分是 D2 最核心的直觉。

玩家每一步的 reward 大致由三部分组成：

### 1）位置奖励

玩家越靠近某个“目标位置”，奖励越高。

如果没有手动设置目标点，默认会把地图中心当成有吸引力的位置。

你可以理解成：

- 地图中心有个酒吧
- 大家都想靠过去

所以这就是论文里常说的 **beach bar** 直觉。

### 2）拥挤惩罚

如果当前位置人太多，就会扣分。

也就是说：

- 好地方人人都想去
- 但人太多又会降低收益

这让玩家形成一个典型的均衡权衡：

- 既想接近目标
- 又想避开拥堵

### 3）动作成本

移动本身也要付出代价。

也就是说：

- 频繁移动不是免费的
- 原地不动有时也是一种理性选择

---

## 8. D2 为什么难

如果只是一个人走网格图，其实并不难。

但 D2 难在它是 **大群体 + 分布耦合**：

- 不是一个人移动
- 而是很多人同时移动
- 每个人都要看整体分布
- 整体分布又是所有人共同策略的结果

所以它的难点不是“动作有 5 个”，而是：

> 你的决策会影响人群分布，而人群分布又决定你的收益。

这就是 MFG 的核心复杂性。

---

## 9. 在 EGTA 里，D2 是怎么被使用的

在你的项目里，D2 主要被拿来做 EGTA 实验。

EGTA 的外层流程可以简单记成：

1. 先有一批策略
2. 在这批策略上求一个经验博弈的混合解
3. 根据这个混合解诱导一个分布
4. 对这个分布求 best response
5. 把新 best response 加进策略池
6. 重复

对 D2 来说，这意味着：

- 经验博弈里的每个策略，都会对应一个二维时空分布
- 这些分布决定当前混合策略的好坏

---

## 10. D2 和论文里的 Game Model Learning 有什么关系

这篇论文的重点不是重新定义 D2，而是：

> 在 D2 这种复杂 MFG 上，精确算 utility 太贵了，所以尝试先学一个“游戏模型”来近似 utility。

也就是说，论文的核心问题是：

- 如果我已经有了一批策略
- 我能不能不每次都调用真实环境
- 而是训练一个模型来预测  
  “某个策略在某个 mixed strategy / induced distribution 下的 payoff”

答案是：

- 可以
- 而且在 D2 上效果还不错

---

## 11. D2 在论文里的直觉

把 D2 放到论文语境里，可以这样理解：

- 真实 D2 环境 = 慢但准确的老师
- 学出来的 game model = 快很多的学生

做法是：

1. 先用真实 D2 环境采样很多数据
2. 数据形式是“策略 + 混合方式 -> utility”
3. 训练一个回归模型去拟合它
4. 之后内层求解时，很多 utility 直接由模型预测

所以论文并不是想替代整个 D2 环境，而是：

**用学习模型来替代大量昂贵的 payoff 查询。**

---

## 12. 你项目里的 D2 数据生成流程

相关文件在：

- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\games\mfg\EGTA\model_learning\generate_data.py`
- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\games\mfg\EGTA\model_learning\sample_utility.py`

这条流程大致是：

1. 先跑 EGTA，得到当前策略池
2. 采样很多 mixed strategy 权重
3. 用这些 mixed strategy 诱导 distribution
4. 再计算每个 pure strategy 在这个 distribution 下的 utility
5. 把这些样本存下来，拿去训练模型

---

## 13. D2 的两种编码思路

你项目里其实有两种理解输入的方式。

### 粗编码（coarse encoding）

输入只保留：

- 当前是哪条纯策略
- 当前 mixed strategy 的权重向量

优点：

- 维度小
- 训练容易
- 很符合论文“压缩输入”的想法

### 细编码（finer encoding）

输入更详细，会把：

- 整个 distribution
- 整个 policy

都格式化成矩阵或时序特征。

优点：

- 信息更全

缺点：

- 维度更大
- 模型训练更重

---

## 14. 你项目里 D2 模型文件是什么

在评估脚本里，2D 对应的模型文件是：

- `E:\Pythoncode\GameTheory\open_spiel\open_spiel\games\mfg\EGTA\model_learning\models\tf_model_crowd2d.h5`

评估脚本会比较两种结果：

1. **真实环境** 推出来的分布
2. **模型辅助** 推出来的分布

如果两者很接近，就说明学到的游戏模型是有用的。

---

## 15. D2 为什么适合做论文实验

因为 D2 兼具这几个特点：

- 状态空间比 1D 更大
- 动作更多
- 分布结构更复杂
- 很直观，容易解释

换句话说：

- 太简单的游戏体现不出 game model learning 的价值
- 太复杂的游戏又不容易验证

而 D2 处在一个很合适的位置：

- 够复杂
- 又还看得懂

---

## 16. 用最通俗的话理解 D2 均衡

如果大家都特别想去中心：

- 中心会变得很挤
- 挤又会被惩罚

于是理性的结果通常不是：

- 所有人都冲向一个点

而是：

- 一部分人靠近中心
- 一部分人在附近绕开
- 形成一种“既接近目标又不过分拥堵”的空间分布

这就是 D2 里最重要的均衡直觉。

---

## 17. 读 D2 代码时最该注意什么

建议重点盯住这三件事：

### 1）状态转移

玩家动作怎么让 `(x, y, t)` 变化。

### 2）分布更新

mean-field 节点怎么把当前整体 distribution 写回状态。

### 3）奖励函数

位置奖励、拥挤惩罚、动作成本是怎么拼起来的。

如果这三件事看明白了，D2 这个游戏你就已经理解大半了。

---

## 18. 一个最简单的脑图

你可以把 D2 想成下面这个流程：

```text
玩家在二维格子上选动作
        ↓
位置变化
        ↓
系统读取当前整体人群分布
        ↓
根据“靠近目标 + 拥挤程度 + 移动代价”计算收益
        ↓
进入下一时刻
```

---

## 19. 和你当前 EGTA 代码怎么连起来

在你的 EGTA 代码里，D2 不是一个独立算法，而是一个 **底层环境**。

关系可以这样记：

- `crowd_modelling_2d.*`：定义 D2 游戏规则
- `egta.py`：做 EGTA 外层循环
- `meta_strategies.py`：做经验博弈上的内层混合求解
- `model_learning/*`：学习一个近似 utility 模型，加速内层求解

所以：

**D2 是“游戏”，EGTA 是“求解框架”，Game Model Learning 是“加速器”。**

---

## 20. 最后一句总结

如果只记一句话，请记这个：

> D2 是一个“大家都想去好位置、但又不想太拥挤”的二维人群移动 Mean Field Game，而论文的目标是学习一个近似游戏模型，减少在这个环境上反复精确计算 utility 的成本。

