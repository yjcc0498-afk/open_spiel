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


_DIRICHLET_TRAIN_PARAMS = np.arange(0.05, 1.05, 0.05)
_DIRICHLET_TEST_PARAMS = np.arange(0.05, 2.05, 0.05)


def _normalize_mixture_weights(weights):
    normalized = np.asarray(weights, dtype=np.float64).copy()
    normalized = np.clip(normalized, 0.0, None)
    total = float(np.sum(normalized, dtype=np.float64))
    if total <= 0.0:
        raise ValueError("Mixed strategy weights must have positive sum.")
    normalized /= total
    if len(normalized) > 1:
        normalized[-1] = max(0.0, 1.0 - float(np.sum(normalized[:-1], dtype=np.float64)))
    normalized /= float(np.sum(normalized, dtype=np.float64))
    return normalized


def _normalize_mixture_matrix(samples, num_policies):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return np.empty((0, num_policies), dtype=np.float64)
    return np.asarray([_normalize_mixture_weights(row) for row in samples], dtype=np.float64)


def _sample_grid_mixtures(num_policies, grid_density, target_count=None):
    full_grid = np.asarray(simplex_grid(m=num_policies, n=grid_density) / grid_density, dtype=np.float64)
    if target_count is None or target_count <= 0 or len(full_grid) <= target_count:
        return _normalize_mixture_matrix(full_grid, num_policies)
    rng = np.random.RandomState(0)
    selected = np.sort(rng.choice(len(full_grid), size=target_count, replace=False))
    return _normalize_mixture_matrix(full_grid[selected], num_policies)


def _sample_dirichlet_mixtures(num_policies, num_samples, test=False):
    if num_samples <= 0:
        return np.empty((0, num_policies), dtype=np.float64)
    params = _DIRICHLET_TEST_PARAMS if test else _DIRICHLET_TRAIN_PARAMS
    samples = []
    base_count = num_samples // len(params)
    remainder = num_samples % len(params)
    for idx, param in enumerate(params):
        count = base_count + (1 if idx < remainder else 0)
        if count <= 0:
            continue
        samples.extend(
            dirichlet_sampling_simplex(
                dim=num_policies,
                num_of_points=count,
                alpha_coef=param))
    return _normalize_mixture_matrix(samples, num_policies)


def _build_sampled_mixtures(num_policies,
                            num_samples,
                            grid,
                            grid_density,
                            test=False,
                            sampling_mode=None,
                            grid_sample_count=None,
                            dirichlet_sample_count=None):
    mode = sampling_mode or ("grid" if grid else "dirichlet")
    if mode == "grid":
        return _sample_grid_mixtures(num_policies, grid_density, grid_sample_count)
    if mode == "dirichlet":
        count = dirichlet_sample_count if dirichlet_sample_count is not None else num_samples
        return _sample_dirichlet_mixtures(num_policies, count, test=test)
    if mode == "hybrid":
        auto_grid_count = grid_sample_count is None
        auto_dirichlet_count = dirichlet_sample_count is None
        if grid_sample_count is None and dirichlet_sample_count is None:
            grid_sample_count = int(np.ceil(num_samples / 2.0))
        elif grid_sample_count is None:
            grid_sample_count = max(num_samples - dirichlet_sample_count, 0)
        elif dirichlet_sample_count is None:
            dirichlet_sample_count = max(num_samples - grid_sample_count, 0)
        grid_samples = _sample_grid_mixtures(num_policies, grid_density, grid_sample_count)
        if auto_dirichlet_count:
            dirichlet_sample_count = max(num_samples - len(grid_samples), 0)
        if auto_grid_count and not auto_dirichlet_count:
            grid_sample_count = max(num_samples - dirichlet_sample_count, 0)
            grid_samples = _sample_grid_mixtures(num_policies, grid_density, grid_sample_count)
        dirichlet_samples = _sample_dirichlet_mixtures(num_policies, dirichlet_sample_count, test=test)
        parts = [part for part in (grid_samples, dirichlet_samples) if len(part) > 0]
        if not parts:
            return np.empty((0, num_policies), dtype=np.float64)
        return _normalize_mixture_matrix(np.concatenate(parts, axis=0), num_policies)
    raise ValueError("Unsupported sampling_mode: {}".format(mode))


class Coarse_Utility_Sampler():
    def __init__(self,
                 mfg_game,
                 egta_solver,
                 num_samples,
                 checkpoint_dir,
                 grid=False,
                 grid_density=4,
                 test=False,
                 encoding="one_hot",
                 sampling_mode=None,
                 grid_sample_count=None,
                 dirichlet_sample_count=None):
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

        # Paper-compatible coarse coding data can use grid, Dirichlet, or a
        # hybrid of both. The same utility_X/Y files are used by MLP and
        # Transformer so architecture is the only controlled variable.
        self._sampled_mixed_policies = _build_sampled_mixtures(
            num_policies=self._num_policies,
            num_samples=num_samples,
            grid=grid,
            grid_density=grid_density,
            test=test,
            sampling_mode=sampling_mode,
            grid_sample_count=grid_sample_count,
            dirichlet_sample_count=dirichlet_sample_count)
        print("Mixed-strategy samples:", len(self._sampled_mixed_policies))

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
                mixed_policy_weights = _normalize_mixture_weights(mixed_policy_weights)
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

