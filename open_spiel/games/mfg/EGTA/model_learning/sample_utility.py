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

from open_spiel.games.mfg.EGTA.utils import dirichlet_sampling_simplex
from open_spiel.games.mfg.EGTA.model_learning.simplex_grid import simplex_grid
from open_spiel.games.mfg.EGTA.model_learning.finer_format_data import Formattor
from open_spiel.games.mfg.EGTA.model_learning.strategy_feature_extractor import StrategyFeatureExtractor
from open_spiel.python.mfg.algorithms.fictitious_play import MergedPolicy
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value

import numpy as np

class Coarse_Utility_Sampler():
    def __init__(self,
                 mfg_game,
                 egta_solver,
                 num_samples,
                 checkpoint_dir,
                 grid=False,
                 grid_density=4,
                 test=False,
                 encoding="one_hot"):
        """
        Sample utility function u(s, mu).
        :param mfg_game: mean field game.
        :param egta_solver: an EGTA solver.
        :param num_samples: The number of samples of mixed strategies to induce mu.
        :param grid: grid sampling
        """
        self._mfg_game = mfg_game
        self._egta_solver = egta_solver
        self._policies = egta_solver._policies
        self._distributions = egta_solver._distributions
        self._num_policies = len(egta_solver._policies)
        self._checkpoint_dir = checkpoint_dir
        self._encoding = encoding

        self._root_states = mfg_game.new_initial_states()
        self._num_players = mfg_game.num_players()
        self._feature_extractor = None
        self._policy_features = None
        # 新增：在 coarse data 生成阶段支持 transformer_stats 编码。
        # 如果开启该模式，会先为所有 pure policy 预计算 [T, 6] 特征，
        # 并把它们和 mixed strategy 一起保存，供后续 Transformer 模型训练使用。
        if self._encoding == "transformer_stats":
            self._feature_extractor = StrategyFeatureExtractor(
                mfg_game, sequence_length=mfg_game.max_game_length())
            self._policy_features = self._feature_extractor.batch_extract(self._policies)

        # Approximately and uniformly sample a collection of mixed strategies.
        if grid:
            self._sampled_mixed_policies = simplex_grid(m=self._num_policies, n=grid_density) / grid_density
        elif test:
            # Hard coded range of sampled test coefficients.
            params = np.arange(0.05, 2.05, 0.05)
            samples = []
            num_samples_per_param = int(num_samples / len(params))
            for param in params:
                samples += list(dirichlet_sampling_simplex(dim=self._num_policies, num_of_points=num_samples_per_param,
                                                           alpha_coef=param))
            self._sampled_mixed_policies = np.array(samples)
        else:
            # Hard coded range of sampled coefficients.
            # params = np.arange(0.05, 2.05, 0.05)
            params = np.arange(0.05, 1.05, 0.05)
            # params = np.arange(0.01, 2.01, 0.1)
            print("Dirichlet params:", params)
            samples = []
            num_samples_per_param = int(num_samples / len(params))
            for param in params:
                samples += list(dirichlet_sampling_simplex(dim=self._num_policies, num_of_points=num_samples_per_param, alpha_coef=param))
            self._sampled_mixed_policies = np.array(samples)

        # Results container.
        self._samples_X = []
        self._samples_norm_X = []
        self._samples_Y = []
        self._samples_features = []
        self._samples_mixed_weights = []

    def compute_utility(self):
        """
        Compute the utility of pure strategy against mean field induced by the sampled mixed strategies.
        """
        for idx, policy in enumerate(self._policies):
            for i, mixed_policy_weights in enumerate(self._sampled_mixed_policies):
                cur_policy = MergedPolicy(self._mfg_game,
                                          list(range(self._mfg_game.num_players())),
                                          self._policies,
                                          self._distributions,
                                          mixed_policy_weights).to_tabular()

                distrib = distribution.DistributionPolicy(self._mfg_game, cur_policy)
                pol_value = policy_value.PolicyValue(self._mfg_game, distrib, policy)
                self.format_data(policy_idx=idx,
                                 mixed_policy_weights=mixed_policy_weights,
                                 pol_value=pol_value)

                # Save data periodically.
                if i % 100 == 0:
                    self.save_data()

            self.save_data()


    def format_data(self, policy_idx, mixed_policy_weights, pol_value):
        """
        Create dataset.
        :param policy_idx: pure strategy index.
        :param mixed_policy_weights: the mixed strategy corresponding to the distribution,
        :param pol_value: utility.
        :return:
        """
        one_hot_policy = np.zeros(self._num_policies)
        one_hot_policy[policy_idx] = 1
        X = np.append(one_hot_policy, mixed_policy_weights)
        weights_std = np.std(mixed_policy_weights)
        if weights_std == 0:
            weights_std = 1.0
        norm_weights = (mixed_policy_weights - np.mean(mixed_policy_weights)) / weights_std
        norm_X = np.append(one_hot_policy, norm_weights)

        # Only work for single-population MFG.
        policy_values = [[] for _ in range(self._num_players)]
        for player, state in enumerate(self._root_states):
            policy_values[player].append(pol_value.value(state))

        self._samples_X.append(X)
        self._samples_norm_X.append(norm_X)
        self._samples_Y.append(policy_values[0])
        if self._encoding == "transformer_stats":
            # 新增：Transformer 路径除了保留旧的 one-hot CSV 外，
            # 还额外保存策略时序特征和 mixed weights，便于新模型直接读取。
            self._samples_features.append(self._policy_features[policy_idx])
            self._samples_mixed_weights.append(mixed_policy_weights)


    def save_data(self):
        """
        Save generated data.
        """
        np.savetxt(self._checkpoint_dir + "/utility_X.csv", np.array(self._samples_X), fmt='%s', delimiter=",")
        np.savetxt(self._checkpoint_dir + "/utility_norm_X.csv", np.array(self._samples_norm_X), fmt='%s', delimiter=",")
        np.savetxt(self._checkpoint_dir + "/utility_Y.csv", np.array(self._samples_Y), fmt='%s', delimiter=",")
        if self._encoding == "transformer_stats":
            np.save(self._checkpoint_dir + "/utility_strategy_features.npy", np.array(self._samples_features))
            np.save(self._checkpoint_dir + "/utility_mixed_weights.npy", np.array(self._samples_mixed_weights))
            np.save(self._checkpoint_dir + "/policy_features.npy", np.array(self._policy_features))


class Finer_Utility_Sampler():
    def __init__(self,
                 mfg_game,
                 egta_solver,
                 num_samples,
                 checkpoint_dir,
                 size,
                 horizon,
                 grid=False,
                 grid_density=4,
                 test=False):
        """
        Sample utility function u(s, mu).
        :param mfg_game: mean field game.
        :param egta_solver: an EGTA solver.
        :param num_samples: The number of samples of mixed strategies to induce mu.
        :param grid: grid sampling
        """
        self._mfg_game = mfg_game
        self._egta_solver = egta_solver
        self._policies = egta_solver._policies
        self._distributions = egta_solver._distributions
        self._num_policies = len(egta_solver._policies)
        self._checkpoint_dir = checkpoint_dir

        self._root_states = mfg_game.new_initial_states()
        self._num_players = mfg_game.num_players()

        self._formattor = Formattor(mfg_game, size=size, horizon=horizon)

        # Approximately and uniformly sample a collection of mixed strategies.
        if grid:
            self._sampled_mixed_policies = simplex_grid(m=self._num_policies, n=grid_density) / grid_density
        elif test:
            # Hard coded range of sampled test coefficients.
            params = np.arange(0.05, 2.05, 0.05)
            samples = []
            num_samples_per_param = int(num_samples / len(params))
            for param in params:
                samples += list(dirichlet_sampling_simplex(dim=self._num_policies, num_of_points=num_samples_per_param,
                                                           alpha_coef=param))
            self._sampled_mixed_policies = np.array(samples)
        else:
            # Hard coded range of sampled coefficients.
            params = np.arange(0.05, 1.05, 0.05)
            # params = np.arange(0.01, 2.01, 0.1)
            print("Dirichlet params:", params)
            samples = []
            num_samples_per_param = int(num_samples / len(params))
            for param in params:
                samples += list(dirichlet_sampling_simplex(dim=self._num_policies, num_of_points=num_samples_per_param, alpha_coef=param))
            self._sampled_mixed_policies = np.array(samples)

        # Results container.
        self._samples_X = []
        self._samples_Y = []

    def compute_utility(self):
        """
        Compute the utility of pure strategy against mean field induced by the sampled mixed strategies.
        """
        for idx, policy in enumerate(self._policies):
            for i, mixed_policy_weights in enumerate(self._sampled_mixed_policies):
                cur_policy = MergedPolicy(self._mfg_game,
                                          list(range(self._mfg_game.num_players())),
                                          self._policies,
                                          self._distributions,
                                          mixed_policy_weights).to_tabular()

                distrib = distribution.DistributionPolicy(self._mfg_game, cur_policy)
                pol_value = policy_value.PolicyValue(self._mfg_game, distrib, policy)
                self.format_data(policy=policy,
                                 distribution=distrib,
                                 pol_value=pol_value)

                # Save data periodically.
                if i % 100 == 0:
                    self.save_data()

            self.save_data()


    def format_data(self, policy, distribution, pol_value):
        """
        Create dataset.
        :param policy_idx: pure strategy index.
        :param mixed_policy_weights: the mixed strategy corresponding to the distribution,
        :param pol_value: utility.
        :return:
        """
        X = self._formattor.formatting(policy, distribution)

        # Only work for single-population MFG.
        policy_values = [[] for _ in range(self._num_players)]
        for player, state in enumerate(self._root_states):
            policy_values[player].append(pol_value.value(state))

        self._samples_X.append(X)
        self._samples_Y.append(policy_values[0])


    def save_data(self):
        """
        Save generated data.
        """
        np.save(self._checkpoint_dir + "/utility_X_finer.npy", np.array(self._samples_X))
        np.save(self._checkpoint_dir + "/utility_Y_finer.npy", np.array(self._samples_Y))

