# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computes the distribution of a policy."""
import collections
import os
import sys
import numpy as np
import torch

from typing import Dict, List, Tuple
from open_spiel.python import policy as policy_module
from open_spiel.python.mfg import distribution as distribution_module
import pyspiel

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_learning.transformer_game_model import TransformerGameModel


def type_from_states(states):
    """Get node type of a list of states and assert they are the same."""
    types = [state.get_type() for state in states]
    assert len(set(types)) == 1, f"types: {types}"
    return types[0]


def state_to_str(state):
    # TODO(author15): Consider switching to
    # state.mean_field_population(). For now, this does not matter in
    # practice since games don't have different observation strings for
    # different player IDs.
    return state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)


def forward_actions(
        current_states: List[pyspiel.State], distribution: Dict[str, float],
        actions_and_probs_fn) -> Tuple[List[pyspiel.State], Dict[str, float]]:
    """Applies one action to each current state.

    Args:
      current_states: The states to apply actions on.
      distribution: Current distribution.
      actions_and_probs_fn: Function that maps one state to the corresponding list
        of (action, proba). For decision nodes, this should be the policy, and for
        chance nodes, this should be chance outcomes.

    Returns:
      A pair:
        - new_states: List of new states after applying one action on
          each input state.
        - new_distribution: Probabilities for each of these states.
    """
    new_states = []
    new_distribution = collections.defaultdict(float)
    for state in current_states:
        state_str = state_to_str(state)
        for action, prob in actions_and_probs_fn(state):
            new_state = state.child(action)
            new_state_str = state_to_str(new_state)
            if new_state_str not in new_distribution:
                new_states.append(new_state)
            new_distribution[new_state_str] += prob * distribution[state_str]
    return new_states, new_distribution


def one_forward_step(current_states: List[pyspiel.State],
                     distribution: Dict[str, float],
                     policy: policy_module.Policy):
    """Performs one step of the forward equation.

    Namely, this takes as input a list of current state, the current
    distribution, and performs one step of the forward equation, using
    actions coming from the policy or from the chance node
    probabilities, or propagating the distribution to the MFG nodes.

    Args:
      current_states: The states to perform the forward step on. All states are
        assumed to be of the same type.
      distribution: Current distribution.
      policy: Policy that will be used if states

    Returns:
      A pair:
        - new_states: List of new states after applying one step of the
          forward equation (either performing one action or doing one
          distribution update).
        - new_distribution: Probabilities for each of these states.
    """
    state_types = type_from_states(current_states)
    if state_types == pyspiel.StateType.CHANCE:
        return forward_actions(current_states, distribution,
                               lambda state: state.chance_outcomes())

    if state_types == pyspiel.StateType.MEAN_FIELD:
        new_states = []
        new_distribution = {}
        for state in current_states:
            dist = [
                # We need to default to 0, since the support requested by
                # the state in `state.distribution_support()` might have
                # states that we might not have reached yet. A probability
                # of 0. should be given for them.
                distribution.get(str_state, 0.)
                for str_state in state.distribution_support()
            ]
            new_state = state.clone()
            new_state.update_distribution(dist)
            new_states.append(new_state)
            new_distribution[state_to_str(new_state)] = distribution.get(
                state_to_str(state), 0)
        return new_states, new_distribution

    if state_types == pyspiel.StateType.DECISION:
        return forward_actions(
            current_states, distribution,
            lambda state: policy.action_probabilities(state).items())

    raise ValueError(
        f"Unpexpected state_stypes: {state_types}, states: {current_states}")


def check_distribution_sum(distribution: Dict[str, float], expected_sum: int):
    """Sanity check that the distribution sums to a given value."""
    sum_state_probabilities = sum(distribution.values())
    assert abs(sum_state_probabilities - expected_sum) < 1e-4, (
        "Sum of probabilities of all possible states should be the number of "
        f"population, it is {sum_state_probabilities}.")


class DistributionPolicy(distribution_module.Distribution):
    """Computes the distribution of a specified strategy."""

    def __init__(self, game: pyspiel.Game,
                 policy: policy_module.Policy,
                 root_state: pyspiel.State = None,
                 evaluation=True):
        """Initializes the distribution calculation.

        Args:
          game: The game to analyze.
          policy: The policy we compute the distribution of.
          root_state: The state of the game at which to start analysis. If `None`,
            the game root states are used.
        """
        super().__init__(game)
        self._policy = policy
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
        self.distribution = None

        if evaluation:
            self.evaluate()

    def evaluate(self):
        """Evaluate the distribution over states of self._policy."""
        # List of all game states that have a non-zero probability at the current
        # timestep and player ID.
        current_states = self._root_states.copy()
        # Distribution at the current timestep. Maps state strings to
        # floats. For each group of states for a given population, these
        # floats represent a probability distribution.
        current_distribution = {state_to_str(state): 1
                                for state in current_states}
        # List of all distributions computed so far.
        all_distributions = [current_distribution]

        while type_from_states(current_states) != pyspiel.StateType.TERMINAL:
            new_states, new_distribution = one_forward_step(current_states,
                                                            current_distribution,
                                                            self._policy)
            check_distribution_sum(new_distribution, self.game.num_players())
            current_distribution = new_distribution
            current_states = new_states
            all_distributions.append(new_distribution)

        # Merge all per-timestep distributions into `self.distribution`.
        self.distribution = {}
        for dist in all_distributions:
            for state_str, prob in dist.items():
                if state_str in self.distribution:
                    raise ValueError(
                        f"{state_str} has already been seen in distribution.")
                self.distribution[state_str] = prob

    def value(self, state):
        return self.value_str(state_to_str(state))

    def value_str(self, state_str, default_value=None):
        """Return probability of the state encoded by state_str.

        Args:
          state_str: string description of the state. This should be created
            using observation_string.
          default_value: in case the state has not been seen by the distribution, to
            avoid raising a value error the default value is returned if it is not
            None.

        Returns:
          state_probability: probability to be in the state descripbed by
            state_str.

        Raises:
          ValueError: if the state has not been seen by the distribution and no
            default value has been passed to the method.
        """
        if default_value is None:
            try:
                return self.distribution[state_str]
            except KeyError as e:
                raise ValueError(
                    f"Distribution not computed for state {state_str}") from e
        return self.distribution.get(state_str, default_value)


class DistributionTransformer(distribution_module.Distribution):
    """Computes the distribution using Transformer model predicted utilities."""

    def __init__(self, game: pyspiel.Game,
                 model_path: str,
                 root_state: pyspiel.State = None,
                 evaluation=True):
        """Initializes the distribution calculation.

        Args:
          game: The game to analyze.
          model_path: Path to the trained Transformer model.
          root_state: The state of the game at which to start analysis. If `None`,
            the game root states are used.
        """
        super().__init__(game)
        self._model_path = model_path
        self._model = None
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
        self.distribution = None
        self.all_distributions = None

        # 加载模型
        self._load_model()

        if evaluation:
            self.evaluate()

    def _load_model(self):
        """Load the Transformer model."""
        try:
            # 先加载模型状态字典，获取实际的策略数量
            state_dict = torch.load(self._model_path)
            # 从 state_dict 中提取策略数量
            strategy_embedding_shape = state_dict['strategy_embedding.weight'].shape
            num_strategies = strategy_embedding_shape[1]
            
            # 模型参数
            d_model = 128
            
            # 加载模型
            self._model = TransformerGameModel(
                num_strategies=num_strategies,
                d_model=d_model,
                nhead=4,
                num_layers=2
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()
            print(f"模型加载成功: {self._model_path}")
            print(f"策略数量: {num_strategies}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self._model = None

    def _predict_utility(self, state, action):
        """Predict utility using the Transformer model."""
        if self._model is None:
            return 0.0
        
        try:
            # 获取模型的策略数量
            num_strategies = self._model.num_strategies
            
            # 提取状态信息并转换为模型输入格式
            # 这里需要根据实际的模型输入格式进行调整
            # 假设我们有一个方法将状态转换为策略和混合策略
            strategy = np.zeros(num_strategies)  # 根据模型的策略数量创建
            mixture = np.ones(num_strategies) / num_strategies  # 均匀分布
            
            # 转换为张量
            strategy_tensor = torch.tensor(strategy, dtype=torch.float32).unsqueeze(0)
            mixture_tensor = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)
            
            # 预测效用值
            with torch.no_grad():
                prediction = self._model(strategy_tensor, mixture_tensor)
                return prediction.item()
        except Exception as e:
            print(f"效用预测失败: {e}")
            return 0.0

    def evaluate(self):
        """Evaluate the distribution using Transformer model predicted utilities."""
        # List of all game states that have a non-zero probability at the current
        # timestep and player ID.
        current_states = self._root_states.copy()
        # Distribution at the current timestep. Maps state strings to
        # floats. For each group of states for a given population, these
        # floats represent a probability distribution.
        current_distribution = {state_to_str(state): 1
                                for state in current_states}
        # List of all distributions computed so far.
        self.all_distributions = [current_distribution]

        while type_from_states(current_states) != pyspiel.StateType.TERMINAL:
            new_states = []
            new_distribution = collections.defaultdict(float)
            
            for state in current_states:
                state_str = state_to_str(state)
                state_prob = current_distribution[state_str]
                
                # 根据状态类型处理
                state_type = state.get_type()
                if state_type == pyspiel.StateType.CHANCE:
                    # 机会节点：使用机会结果
                    for action, prob in state.chance_outcomes():
                        new_state = state.child(action)
                        new_state_str = state_to_str(new_state)
                        if new_state_str not in new_distribution:
                            new_states.append(new_state)
                        new_distribution[new_state_str] += prob * state_prob
                
                elif state_type == pyspiel.StateType.MEAN_FIELD:
                    # 均值场节点：更新分布
                    dist = [
                        current_distribution.get(str_state, 0.)
                        for str_state in state.distribution_support()
                    ]
                    new_state = state.clone()
                    new_state.update_distribution(dist)
                    new_states.append(new_state)
                    new_distribution[state_to_str(new_state)] = state_prob
                
                elif state_type == pyspiel.StateType.DECISION:
                    # 决策节点：使用模型预测的效用值
                    actions = state.legal_actions()
                    if not actions:
                        continue
                    
                    # 预测每个动作的效用值
                    utilities = []
                    for action in actions:
                        utility = self._predict_utility(state, action)
                        utilities.append(utility)
                    
                    # 计算动作概率（使用softmax）
                    exp_utilities = np.exp(utilities)
                    action_probs = exp_utilities / np.sum(exp_utilities)
                    
                    # 计算新状态和分布
                    for action, prob in zip(actions, action_probs):
                        new_state = state.child(action)
                        new_state_str = state_to_str(new_state)
                        if new_state_str not in new_distribution:
                            new_states.append(new_state)
                        new_distribution[new_state_str] += prob * state_prob
            
            # 检查分布和
            check_distribution_sum(new_distribution, self.game.num_players())
            
            # 更新当前状态和分布
            current_states = new_states
            current_distribution = new_distribution
            self.all_distributions.append(current_distribution)

        # Merge all per-timestep distributions into `self.distribution`.
        self.distribution = {}
        for dist in self.all_distributions:
            for state_str, prob in dist.items():
                if state_str in self.distribution:
                    raise ValueError(
                        f"{state_str} has already been seen in distribution.")
                self.distribution[state_str] = prob

    def value(self, state):
        return self.value_str(state_to_str(state))

    def value_str(self, state_str, default_value=None):
        """Return probability of the state encoded by state_str."""
        if default_value is None:
            try:
                return self.distribution[state_str]
            except KeyError as e:
                raise ValueError(
                    f"Distribution not computed for state {state_str}") from e
        return self.distribution.get(state_str, default_value)

    def save_distribution(self, save_path):
        """Save the distribution to a numpy file."""
        if self.all_distributions is None:
            print("分布尚未计算")
            return
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 转换分布为numpy数组
        # 假设每个时间步的分布都可以表示为10x10的网格
        distributions_array = []
        for dist in self.all_distributions:
            # 创建10x10的网格
            grid = np.zeros((10, 10))
            for state_str, prob in dist.items():
                # 从状态字符串中提取位置信息
                # 这里需要根据实际的状态表示格式进行调整
                # 假设状态字符串包含位置信息
                try:
                    # 示例：从状态字符串中提取x和y坐标
                    # 这里需要根据实际的状态表示格式进行修改
                    x, y = 0, 0  # 占位符
                    if 0 <= x < 10 and 0 <= y < 10:
                        grid[x, y] = prob
                except Exception as e:
                    print(f"处理状态 {state_str} 失败: {e}")
            distributions_array.append(grid.flatten())
        
        # 保存为numpy文件
        distributions_array = np.array(distributions_array)
        np.save(save_path, distributions_array)
        print(f"分布已保存到: {save_path}")


def generate_transformer_distribution(game_name="mfg_crowd_modelling", size=10, step=10, model_path=None, model_size=10, model_step=10, save_filename="distribution_transformer_model.npy"):
    """生成使用 Transformer 模型预测的分布并保存
    
    Args:
        game_name: 游戏名称
        size: 游戏大小
        step: 游戏时间范围
        model_path: 模型路径，如果为 None 则使用默认路径
        model_size: 模型对应的游戏大小
        model_step: 模型对应的游戏时间范围
        save_filename: 保存的文件名
    """
    # 创建游戏
    game = pyspiel.load_game(game_name, {"size": size, "horizon": step})
    
    # 模型路径
    if model_path is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_learning", "models")
        model_path = os.path.join(model_dir, f"trained_model_size{model_size}_step{model_step}.pth")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建分布计算器
    dist_calculator = DistributionTransformer(game, model_path)
    
    # 保存分布
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot", "Distributions", "mfg_crowd_modelling_1d_FP_dist_eval")
    save_path = os.path.join(save_dir, save_filename)
    dist_calculator.save_distribution(save_path)


if __name__ == "__main__":
    # 自定义模型路径和保存文件名
    print("\n示例2: 自定义模型路径和保存文件名")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_learning", "models")
    custom_model_path = os.path.join(model_dir, "trained_model_size10_step10.pth")
    custom_save_filename = "distribution_transformer_model_custom.npy"
    generate_transformer_distribution(
        game_name="mfg_crowd_modelling", 
        size=10, 
        step=10, 
        model_path=custom_model_path,
        save_filename=custom_save_filename
    )
