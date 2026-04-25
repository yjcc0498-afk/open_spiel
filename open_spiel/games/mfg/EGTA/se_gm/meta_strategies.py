# Meta-strategy solvers are implemented with models in this file

import numpy as np
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms.fictitious_play import MergedPolicy
from open_spiel.games.mfg.EGTA.utils import dirichlet_sampling_simplex

from open_spiel.games.mfg.EGTA.meta_strategies import ILMergedPolicy
from open_spiel.games.mfg.EGTA.meta_strategies import MetaStrategyMethod
from open_spiel.games.mfg.EGTA.se_gm.utils import qp_projection2

from scipy.stats import wasserstein_distance


class FictitiousPlayMSS(MetaStrategyMethod):
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions,
                 num_iterations,
                 EGTA_iterations,
                 model,
                 w_distance,
                 planning_iters,
                 fine_tune_iters,
                 root_state=None,
                 regret_threshold=0.02,
                 norm=False,
                 num_samples=800):
        """
        Apply fictitious player on the empirical game.
        :param mfg_game: The full game to analyze.
        :param policies: A `List[policy_std.Policy]` object.
        :param distributions: A `List[distribution_std.Distribution]` object corresponding to the policies.
        :param num_iterations: Number of inner loop iterations.
        :param meta_strategy_method: a object of Meta-strategy solver class.
        """
        super(FictitiousPlayMSS, self).__init__(mfg_game, policies, distributions)
        self._num_iterations = num_iterations
        self._regret_threshold = regret_threshold
        self._EGTA_iterations = EGTA_iterations
        self._planning_iters = planning_iters
        self._fine_tune_iters = fine_tune_iters
        self._w_distance = w_distance #TODO: Maybe it is hard to set a uniform threshold.

        # Switch for using w_distance or fixed number of iteration to swap planning and simulating.
        # False: w_distance True: simulating.
        self._switch = False

        #Model-based Learning
        self._model = model
        self._norm = norm
        self._num_samples = num_samples
        # 新增：当使用 Transformer 编码时，进入 inner loop 之前先为当前策略池
        # 预计算所有 pure policy 的时序特征，后续 predict 直接查缓存。
        if self._model is not None:
            self._model.refresh_policy_features(self._mfg_game, self._policies)

        if root_state is None:
            self._root_states = mfg_game.new_initial_states()
        else:
            self._root_states = [root_state]

        # Initialize with uniform policy, assuming uniform policy exists in the empirical game.
        self._updated_policy = policy_std.UniformRandomPolicy(self._mfg_game)
        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        # Current number of iterations.
        self._current_iter = 0

        # The frequency of each policy being a best response in the empirical game, one frequency array per player.
        self._policy_frequency_counter = [np.zeros(len(self._policies)) for _ in range(self._num_players)]
        for player in range(self._num_players):
            self._policy_frequency_counter[player][0] += 1 # increment for initial uniform strategy


    def update_weights(self):
        """
        Calculate the fictitious play weights at current iteration.
        Then previous merged policy and new best-response policy are weighted.
        :return: weights on merged policy and new best-response policy.
        """
        _fp_step = self.get_current_iter()
        self._weights = [1.0*_fp_step/(_fp_step+1), 1.0/(_fp_step+1)]
        return self._weights

    def reset(self):
        """
        Reset the current policies.
        """
        # Initialize with uniform policy.
        self._updated_policy = policy_std.UniformRandomPolicy(self._mfg_game)
        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)
        self._current_iter = 0
        self._policy_frequency_counter = [np.zeros(len(self._policies)) for _ in range(self._num_players)]
        for player in range(self._num_players):
            self._policy_frequency_counter[player][0] += 1


    def iteration(self, use_simulator=False):
        """
        One iteration of inner loop.
        :return:
        """
        # Find best-response policy within the empirical game.
        policy_values = [[] for _ in range(self._num_players)]
        model_policy_values = [[] for _ in range(self._num_players)]

        for idx, policy in enumerate(self._policies):
            model_pol_value = self.predict(idx)
            pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, policy)
            for player, state in enumerate(self._root_states):
                model_policy_values[player].append(model_pol_value) # ONLY WORK FOR SINGLE POPULATION.
                policy_values[player].append(pol_value.value(state))
                # Save data for all s if use_simulator or only the recently added s if not.
                if use_simulator or idx == len(self._policies) - 1:
                    self.save_new_data(idx, policy_values[player][-1]) # ONLY WORK FOR SINGLE POPULATION.

        # Construct new policy values: t-1 using model, t using simulator.
        # TODO: check if running the warm-started model works directly or need to use true utilities for the last strategy.
        for player, _ in enumerate(self._root_states):
            if use_simulator:
                model_policy_values[player] = policy_values[player]
            else:
                model_policy_values[player][-1] = policy_values[player][-1]

        model_br_policy_indice = np.argmax(model_policy_values, axis=1)
        br_policy_indice = np.argmax(policy_values, axis=1)

        model_br_policy_per_player = []
        model_br_distribution_per_player = []

        for player in range(self._num_players):
            model_br_idx = model_br_policy_indice[player]
            model_br_policy_per_player.append(self._policies[model_br_idx])
            model_br_distribution_per_player.append((self._distributions[model_br_idx])) # note why has () inside. No effect.
            self._policy_frequency_counter[player][model_br_idx] += 1


        br_policy_value_per_player = []

        for player in range(self._num_players):
            br_idx = br_policy_indice[player]
            br_policy_value_per_player.append(policy_values[player][br_idx])

        # Compute regret with respect to the empirical games.
        regret = self.regret(sum(br_policy_value_per_player))

        self._updated_policy = ILMergedPolicy(game=self._mfg_game,
                                                  player_ids=list(range(self._num_players)),
                                                  prev_policy=self._updated_policy,
                                                  prev_distribution=self._updated_policy_distribution,
                                                  br_policies=model_br_policy_per_player,
                                                  br_distributions=model_br_distribution_per_player,
                                                  weights=self.update_weights()).to_tabular()



        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        return regret

    def run(self):
        """
        Run fictitious play.
        """
        if len(self._policies) == 1:
            pass


        self.warm_start()

        regret_list = []
        period = self._planning_iters + self._fine_tune_iters
        for i in range(self._num_iterations):
            if i % period < self._planning_iters:
                regret = self.iteration(use_simulator=False)
            else:
                regret = self.iteration(use_simulator=True)
            regret_list.append(regret)

            if i % period == self._planning_iters - 1:
                print("Model regret:", regret_list)

            # TODO: for each period, fine-tune the model. Make sure the new data is set to be None.
            if i % period == period - 1:
                self._model.combine_new_old_data()
                self._model.fit()

            self._current_iter += 1
            if regret < self._regret_threshold:
                break
        print("Inner loop regret:", regret_list)


    def predict(self, policy_idx):
        """ Using model to predict the utility given s and mu."""
        if self._model is None:
            raise ValueError("The model has not been specified.")
        # 新增：这里不再手工拼 one-hot。
        # 改为统一走 _build_model_input，让 one_hot / transformer_stats 两种编码
        # 共用同一套 inner-loop 调用逻辑。
        cur_weights = self.get_weights_on_orig_policies()[0]
        X = self._build_model_input(policy_idx, cur_weights)
        estimate_Y = np.squeeze(self._model.predict(X))[()]

        return estimate_Y

    def save_new_data(self, idx, value):
        """ Combined newly generated data with new_data in the model."""
        cur_weights = self.get_weights_on_orig_policies()[0]
        X = self._build_model_input(idx, cur_weights)
        Y = np.array([[value]])
        if self._model._new_X is None:
            self._model.update_dataset(new_X=X, new_Y=Y)
        else:
            self._model.combine_new_data(new_X=X, new_Y=Y)


    def regret(self, deviation_payoff):
        """
        Calculate the regret of the self._updated_policy within the empirical game.
        :param deviation_payoff: A sum of players' deviation payoff.
        :return:
        """
        pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, self._updated_policy)
        regret = deviation_payoff - sum([pol_value.value(state) for state in self._root_states])

        return regret


    def warm_start(self):
        """ Sample some data for the newly added strategy. """
        params = np.arange(0.05, 1.05, 0.05)
        samples = []
        num_samples_per_param = int(self._num_samples / len(params))
        if num_samples_per_param == 0:
            num_samples_per_param = 1
        for param in params:
            samples += list(dirichlet_sampling_simplex(dim=len(self._policies), num_of_points=num_samples_per_param,
                                                       alpha_coef=param))
        sampled_mixed_policies = np.array(samples)

        # Results container.

        self._samples_X = []
        self._samples_Y = []

        for i, mixed_policy_weights in enumerate(sampled_mixed_policies):
            cur_policy = MergedPolicy(self._mfg_game,
                                      list(range(self._mfg_game.num_players())),
                                      self._policies,
                                      self._distributions,
                                      mixed_policy_weights).to_tabular()

            distrib = distribution.DistributionPolicy(self._mfg_game, cur_policy)
            pol_value = policy_value.PolicyValue(self._mfg_game, distrib, self._policies[-1])
            self.format_data(policy_idx=len(self._policies)-1,
                             mixed_policy_weights=mixed_policy_weights,
                             pol_value=pol_value)


        print('self._samples_X:', self._samples_X)

        if self._model._new_X is None:
            print('enter here1')
            self._model.update_dataset(new_X=self._samples_X, new_Y=self._samples_Y)
        else:
            print('enter here2')
            self._model.combine_new_data(new_X=self._samples_X, new_Y=self._samples_Y)

        print("warm-start:", self._model._new_X)

        # Only generate new data rather than fine tune the model.


    def format_data(self, policy_idx, mixed_policy_weights, pol_value):
        """
        Create dataset.
        :param policy_idx: pure strategy index.
        :param mixed_policy_weights: the mixed strategy corresponding to the distribution,
        :param pol_value: utility.
        :return:
        """
        # 新增：warm-start / fine-tune 阶段采样的数据也统一通过新输入接口构造，
        # 这样训练数据格式能与推理格式保持一致。
        X = self._build_model_input(policy_idx, mixed_policy_weights)

        # Only work for single-population MFG.
        policy_values = [[] for _ in range(self._num_players)]
        for player, state in enumerate(self._root_states):
            policy_values[player].append(pol_value.value(state))

        self._samples_X.append(X)
        self._samples_Y.append([policy_values[0]])

    def _pad_mixed_weights(self, mixed_weights):
        padded = np.zeros(self._EGTA_iterations + 1, dtype=np.float32)
        mixed_weights = np.asarray(mixed_weights, dtype=np.float32)
        padded[:len(mixed_weights)] = mixed_weights
        return padded

    def _build_model_input(self, policy_idx, mixed_weights):
        # 统一把“pure strategy + mixed strategy”转换成模型输入。
        # one-hot 模式返回扁平向量；
        # transformer_stats 模式返回双输入字典，并补上 batch 维度。
        padded_weights = self._pad_mixed_weights(mixed_weights)
        X = self._model.build_sample_input(policy_idx, padded_weights)
        if self._model.encoding == 'transformer_stats':
            return {
                "strategy_features": X["strategy_features"][None, ...],
                "mixed_weights": X["mixed_weights"][None, ...],
            }
        return X[None, ...]


    def get_output_policies(self):
        """
        Only valid after run_inner_loop. Otherwise return initial self._updated_policies.
        :return: the output policies.
        """
        return self._updated_policy

    def get_policies(self):
        """
        Get empirical game policies.
        :return: Empirical game policies.
        """
        return self._policies

    def get_weights_on_merged_policies(self):
        return self._weights

    def get_weights_on_orig_policies(self):
        """
        Get weights on the self._policies.
        :return:
        """
        counter = self._policy_frequency_counter.copy()
        for player in range(self._num_players):
            counter[player] = counter[player] / np.sum(counter[player])

        return counter

    def get_current_iter(self):
        """
        :return: current number of FP iteration.
        """
        return self._current_iter




"""
Replicator Dynamics.
"""


class ReplicatorDynamicsMSS(MetaStrategyMethod):
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions,
                 num_iterations,
                 EGTA_iterations,
                 model,
                 w_distance,
                 planning_iters,
                 fine_tune_iters,
                 root_state=None,
                 dt=0.03,
                 regret_threshold=0.02):
        """
        Apply replicator dynamics to the empirical game.
        ONLY WORK FOR SINGLE POPULATION. (Different players should have different weights).
        :param mfg_game: The full game to analyze.
        :param policies: A `List[policy_std.Policy]` object.
        :param distributions: A `List[distribution_std.Distribution]` object corresponding to the policies.
        :param num_iterations: Number of inner loop iterations.
        """
        super(ReplicatorDynamicsMSS, self).__init__(mfg_game, policies, distributions)
        self._num_iterations = num_iterations
        self._regret_threshold = regret_threshold
        self._dt = dt

        if self._num_players > 1:
            raise NotImplementedError("Currently only work for single population.")

        #Model-based Learning
        self._model = model
        self._EGTA_iterations = EGTA_iterations
        self._planning_iters = planning_iters
        self._fine_tune_iters = fine_tune_iters
        self._w_distance = w_distance
        # 新增：RD 路径和 FP 一样，初始化时先刷新当前策略池的时序特征缓存。
        if self._model is not None:
            self._model.refresh_policy_features(self._mfg_game, self._policies)

        # Switch for using distance or fixed number of iteration to swap planning and simulating.
        self._enable_w_distance = True

        if root_state is None:
            self._root_states = mfg_game.new_initial_states()
        else:
            self._root_states = [root_state]

        # Initialize with uniform over policies, assuming uniform policy exists in the empirical game.
        self._updated_policy = MergedPolicy(
            self._mfg_game, list(range(self._num_players)),
            self._policies, self._distributions,
            [1.0 / len(self._policies) for _ in range(len(self._policies))]
        ).to_tabular()
        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        # Current number of iterations.
        self._current_iter = 0

        # The frequency of each policy being a best response in the empirical game, one frequency array per player.
        self._model_weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
                             range(self._num_players)]


    def reset(self, padding=True, offset=0.005):
        """
        Reset the current policies.
        """
        # # Initialize with uniform policy.
        # self._updated_policy = MergedPolicy(
        #     self._mfg_game, list(range(self._num_players)),
        #     self._policies, self._distributions,
        #     [1.0 / len(self._policies) for _ in range(len(self._policies))]
        # ).to_tabular()
        # self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)
        #
        # # Current number of iterations.
        # self._current_iter = 0
        #
        # # The frequency of each policy being a best response in the empirical game, one frequency array per player.
        # self._model_weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
        #                      range(self._num_players)]


        # Start RD with equilibrium strategy from previous iteration.
        # TODO: check how the preserved distributions affect the model usage.
        self._current_iter = 0
        if len(self._policies) > 1 and padding: # No zero to add at the first iter.
            for player in range(self._num_players):
                self._model_weights[player] = np.pad(self._model_weights[player], (0,1), 'constant')
                self._model_weights[player][-1] = offset
                self._model_weights[player][0] -= offset

    def update_weights(self, weights, values):
        """ Update RD weights. """
        for player in range(self._num_players):
            weights[player] += weights[player] * (np.array(values[player]) - np.sum(weights[player] * np.array(values[player]))) * self._dt
            if np.sum(weights[player]) != 1:
                weights[player] /= np.sum(weights[player])


    def iteration(self, use_simulator=False):
        policy_values = [[] for _ in range(self._num_players)]
        model_policy_values = [[] for _ in range(self._num_players)]

        for idx, policy in enumerate(self._policies):
            model_pol_value = self.predict(idx)
            pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, policy)
            for player, state in enumerate(self._root_states):
                model_policy_values[player].append(model_pol_value)
                policy_values[player].append(pol_value.value(state))
                # Save data for model fine-tuning.
                if use_simulator or idx == len(self._policies) - 1:
                    self.save_new_data(idx, policy_values[player][-1]) # ONLY WORK FOR SINGLE POPULATION.

        # Construct new policy values: t-1 using model, t using simulator.
        for player, _ in enumerate(self._root_states):
            if use_simulator:
                model_policy_values[player] = policy_values[player]
            else:
                # Model will give an arbitrary payoff for the new strategy
                # since the model is only effective for \tau-1 iterations.
                model_policy_values[player][-1] = policy_values[player][-1]


        regret = self.regret(sum(np.max(policy_values, axis=1)))

        self.update_weights(self._model_weights, model_policy_values)

        self._updated_policy = MergedPolicy(
            self._mfg_game, list(range(self._num_players)),
            self._policies, self._distributions,
            self._model_weights[0] # ONLY WORK FOR SINGLE POPULATION.
        ).to_tabular()

        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        return regret

    def run_v1(self):
        """
        Run Replicator Dynamics. Use fixed number of iterations as a criterion to stop planning.
        """
        regret_list = []
        iter_for_model = [] # Record iterations in which model is used.
        norms = []
        num_policies = len(self._policies)

        # for i in range(self._planning_iters * num_policies):
        # Previously, we set model iter to be 30.
        for i in range(20):
            # Using fixed number of iterations as switch criterion.

            cur_weights = self.get_model_weights()[0]
            proj_weights = qp_projection2(cur_weights)
            norm = np.linalg.norm(cur_weights - proj_weights)
            norms.append(norm)
            if norm > self._w_distance:
                break

            regret = self.iteration(use_simulator=False)
            iter_for_model.append(1)
            regret_list.append(regret)
            if i % num_policies == 0:
                self._current_iter += 1
            if regret < self._regret_threshold:
                break

        extra = 30 // num_policies

        for i in range(self._fine_tune_iters + self._planning_iters - extra):
            regret = self.iteration(use_simulator=True)
            iter_for_model.append(0)
            regret_list.append(regret)
            self._current_iter += 1
            if regret < self._regret_threshold:
                break

        print("Model training starts.")
        self._model.combine_new_old_data()
        loss = self._model.fit()
        print("Model training ends with loss:", loss)


        print("Iter_for_model:", iter_for_model, "Total:", np.sum(iter_for_model))
        print("Norms:", norms)
        print("Model regret:", np.array(regret_list) * np.array(iter_for_model))
        print("Inner loop regret:", regret_list)



    def run_v2(self):
        """
        Run Replicator Dynamics. Use distance as a criterion to stop planning.
        """
        print("In v2.")
        regret_list = []
        iter_for_model = [] # Record iterations in which model is used.
        norms = []
        stop_planning = False # Choose whether it is able to switch back to planning.
        period = self._planning_iters + self._fine_tune_iters
        num_policies = len(self._policies)

        i = 0 # Number of simulations.
        j = 0 # Total iters for model using = sum(iter_for_model)
        while i < self._num_iterations:
            # Projection from n-dim to (n-1)-dim.
            cur_weights = self.get_model_weights()[0]
            proj_weights = qp_projection2(cur_weights)

            norm = np.linalg.norm(cur_weights - proj_weights)
            norms.append(norm)
            if norm < self._w_distance and not stop_planning:
                regret = self.iteration(use_simulator=False)
                iter_for_model.append(1) # 1 for planning.

                # Note: using model also needs simulation of the last strategy.
                j += 1
                if j % num_policies == 0:
                    i += 1

            else:
                #TODO: whether to add a protection mechanism to make sure simulating is run.
                stop_planning = True # If switch to simulation, then stay in simulation.
                regret = self.iteration(use_simulator=True)
                iter_for_model.append(0) # 0 for simulating.
                i += 1

            regret_list.append(regret)

            self._current_iter += 1
            # if regret < self._regret_threshold:
            #     break
            if regret < self._regret_threshold or self._current_iter > 4 * period:
                break


        print("Iter_for_model:", iter_for_model, "Total Model iters:", np.sum(iter_for_model), "Total iters:", self._current_iter)
        print("Norms:", norms)
        print("Model training starts.")
        self._model.combine_new_old_data()
        loss = self._model.fit()
        print("Model training ends with loss:", loss)

        print("Inner loop regret:", regret_list)

    def run(self, v2=False):
        """ Control Panel for running inner loop."""
        if v2:
            print("v2 is running.")
            self.run_v2()
        else:
            print("v1 is running.")
            self.run_v1()


    def regret(self, deviation_payoff):
        """
        Calculate the regret of the self._updated_policy within the empirical game.
        :param deviation_payoff: A sum of players' deviation payoff.
        """
        pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, self._updated_policy)
        regret = deviation_payoff - sum([pol_value.value(state) for state in self._root_states])

        return regret

    def predict(self, policy_idx):
        """ Using model to predict the utility given s and mu."""
        if self._model is None:
            raise ValueError("The model has not been specified.")
        # 新增：RD 的 utility 查询同样改成统一输入构造，不再直接依赖 one-hot。
        cur_weights = qp_projection2(self.get_model_weights()[0]) # Always projection.
        X = self._build_model_input(policy_idx, cur_weights)
        estimate_Y = np.squeeze(self._model.predict(X))[()]

        return estimate_Y

    def save_new_data(self, idx, value):
        """ Save data generated in simulation to fine-tune a model."""
        cur_weights = self.get_model_weights()[0]
        X = self._build_model_input(idx, cur_weights)
        Y = np.array([[value]])

        if self._model._new_X is None:
            self._model.update_dataset(new_X=X, new_Y=Y)
        else:
            self._model.combine_new_data(new_X=X, new_Y=Y)

    def _pad_mixed_weights(self, mixed_weights):
        padded = np.zeros(self._EGTA_iterations + 1, dtype=np.float32)
        mixed_weights = np.asarray(mixed_weights, dtype=np.float32)
        padded[:len(mixed_weights)] = mixed_weights
        return padded

    def _build_model_input(self, policy_idx, mixed_weights):
        # 和 FP 版本保持一致，避免两套 inner-loop 对输入格式各自维护一份逻辑。
        padded_weights = self._pad_mixed_weights(mixed_weights)
        X = self._model.build_sample_input(policy_idx, padded_weights)
        if self._model.encoding == 'transformer_stats':
            return {
                "strategy_features": X["strategy_features"][None, ...],
                "mixed_weights": X["mixed_weights"][None, ...],
            }
        return X[None, ...]


    def get_weights(self):
        raise NotImplementedError


    def get_model_weights(self):
        return self._model_weights


    def get_policies(self):
        """
        Get empirical game policies.
        :return: Empirical game policies.
        """
        return self._policies

    def get_output_policies(self):
        """
        Only valid after run().
        :return: the output policies.
        """
        return self._updated_policy

    def get_current_iter(self):
        """
        :return: current number of FP iteration.
        """
        return self._current_iter

    def get_weights_on_orig_policies(self):
        """
        Get weights on the self._policies.
        :return:
        """
        if self._model is not None:
            return self.get_model_weights()
        else:
            return self.get_weights()


MFG_META_STRATEGY_METHODS = {
    "FP": FictitiousPlayMSS,
    "RD": ReplicatorDynamicsMSS
}


# def warm_start(self):
#     """ Sample some data for the newly added strategy and then fine tune the model. """
#     params = np.arange(0.05, 1.05, 0.05)
#     samples = []
#     num_samples_per_param = int(self._num_samples / len(params))
#     for param in params:
#         samples += list(dirichlet_sampling_simplex(dim=len(self._policies), num_of_points=num_samples_per_param,
#                                                    alpha_coef=param))
#     sampled_mixed_policies = np.array(samples)
#
#     # Results container.
#
#     self._samples_X = []
#     self._samples_Y = []
#
#     for idx, policy in enumerate(self._policies):
#         for i, mixed_policy_weights in enumerate(sampled_mixed_policies):
#             cur_policy = MergedPolicy(self._mfg_game,
#                                       list(range(self._mfg_game.num_players())),
#                                       self._policies,
#                                       self._distributions,
#                                       mixed_policy_weights).to_tabular()
#
#             distrib = distribution.DistributionPolicy(self._mfg_game, cur_policy)
#             pol_value = policy_value.PolicyValue(self._mfg_game, distrib, policy)
#             self.format_data(policy_idx=idx,
#                              mixed_policy_weights=mixed_policy_weights,
#                              pol_value=pol_value)
#
#     if self._model._new_X is None:
#         self._model.update_dataset(new_X=self._samples_X, new_Y=self._samples_Y)
#     else:
#         self._model.combine_new_data(new_X=self._samples_X, new_Y=self._samples_Y)
#
#     # self._model.fit()
#
#
# def format_data(self, policy_idx, mixed_policy_weights, pol_value):
#     """
#     Create dataset.
#     :param policy_idx: pure strategy index.
#     :param mixed_policy_weights: the mixed strategy corresponding to the distribution,
#     :param pol_value: utility.
#     :return:
#     """
#     one_hot_policy = np.zeros(self._EGTA_iterations + 1)
#     one_hot_policy[policy_idx] = 1
#     mixed_weights = np.zeros(self._EGTA_iterations + 1)
#     mixed_weights[:len(mixed_policy_weights)] = mixed_policy_weights
#     X = np.append(one_hot_policy, mixed_weights)
#
#     # Only work for single-population MFG.
#     policy_values = [[] for _ in range(self._num_players)]
#     for player, state in enumerate(self._root_states):
#         policy_values[player].append(pol_value.value(state))
#
#     self._samples_X.append(X)
#     self._samples_Y.append(policy_values[0])
