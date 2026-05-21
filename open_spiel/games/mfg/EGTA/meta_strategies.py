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

# Meta-strategy solvers are implemented in this file

from typing import List

import numpy as np
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg.algorithms.fictitious_play import MergedPolicy


class ILMergedPolicy(policy_std.Policy):
  """Merge several policies for fictitious-play inner loop."""

  def __init__(self, game,
               player_ids,
               prev_policy,
               prev_distribution,
               br_policies: List[policy_std.Policy],
               br_distributions: List[distribution_std.Distribution],
               weights: List[float]):
      """Initializes the merged policy.
      Args:
        game: The game to analyze.
        player_ids: list of player ids for which this policy applies; each should
          be in the range 0..game.num_players()-1.
        prev_policy: A merged policy from previous FP iterations.
        br_policies: A `List[policy_std.Policy]` object, one policy (BR within the empirical game) per player.
        br_distributions: A `List[distribution_std.Distribution]` object.
        weights: A `List[float]` object. They should sum to 1.
      """
      super(ILMergedPolicy, self).__init__(game, player_ids)
      self._prev_policy = prev_policy
      self._prev_distribution = prev_distribution
      self._br_policies = br_policies
      self._br_distributions = br_distributions
      self._weights = weights

      assert len(br_policies) == len(br_distributions), (
          f'Length mismatch {len(br_policies)} != {len(br_distributions)}')

  def action_probabilities(self, state, player_id=None):
    """Only choose new to-be-merged policy based on which state population is."""
    if hasattr(state, "_population"):
        # If the game has multiple populations.
        population = state._population
    else:
        # If the game only has single population.
        population = 0

    selected_policy = self._br_policies[population]
    selected_distribution = self._br_distributions[population]

    state_policies = [self._prev_policy, selected_policy]
    state_distributions = [self._prev_distribution, selected_distribution]

    action_prob = []
    legal = state.legal_actions()
    num_legal = len(legal)
    for a in legal:
        merged_pi = 0.0
        norm_merged_pi = 0.0
        for p, d, w in zip(state_policies, state_distributions, self._weights):
            # Use previous tabular policy to avoid large depth of policy callings.
            merged_pi += w * d(state) * p(state)[a]
            norm_merged_pi += w * d(state)
        if norm_merged_pi > 0.0:
            action_prob.append((a, merged_pi / norm_merged_pi))
        else:
            action_prob.append((a, 1.0 / num_legal))
    return dict(action_prob)


class MetaStrategyMethod(object):
    """ Base class for Meta-Strategy Solver. """
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions):
        self._mfg_game = mfg_game
        self._num_players = mfg_game.num_players()
        self._policies = policies
        self._distributions = distributions

    def run(self):
        """ Run the meta-strategy solver. """
        raise NotImplementedError

    def get_output_policies(self):
        """
        :return: the output of the meta-strategy solver.
        """
        raise NotImplementedError

    def reset(self):
        """ Reset the MSS."""
        raise NotImplementedError

class FictitiousPlayMSS(MetaStrategyMethod):
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions,
                 num_iterations,
                 root_state=None,
                 regret_threshold=0.02,
                 model=None,
                 norm=False,
                 resample=False):
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

        #Model-based Learning
        self._model = model
        self._norm = norm
        self._resample = resample

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


        #Test if warm-start brings in benefits.
        # self._current_iter = 0
        # for player in range(self._num_players):
        #     self._policy_frequency_counter[player] = np.pad(self._policy_frequency_counter[player], (0,1), 'constant')


    def iteration(self):
        """
        One iteration of inner loop.
        :return:
        """
        # print("Start of a new iterations.")
        # Find best-response policy within the empirical game.

        policy_values = [[] for _ in range(self._num_players)]
        if self._model is not None:
            model_policy_values = [[] for _ in range(self._num_players)]

        for idx, policy in enumerate(self._policies):
            if self._model is not None:
                model_pol_value = self.predict(idx)
            pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, policy)
            for player, state in enumerate(self._root_states):
                if self._model is not None:
                    model_policy_values[player].append(model_pol_value)
                policy_values[player].append(pol_value.value(state))

        if self._model is not None:
            model_br_policy_indice = np.argmax(model_policy_values, axis=1)
        br_policy_indice = np.argmax(policy_values, axis=1)

        if self._model is not None:
            model_br_policy_per_player = []
            model_br_distribution_per_player = []

            for player in range(self._num_players):
                model_br_idx = model_br_policy_indice[player]
                model_br_policy_per_player.append(self._policies[model_br_idx])
                model_br_distribution_per_player.append((self._distributions[model_br_idx])) # note why has () inside. No effect.
                self._policy_frequency_counter[player][model_br_idx] += 1

        br_policy_per_player = []
        br_policy_value_per_player = []
        br_distribution_per_player = []

        for player in range(self._num_players):
            br_idx = br_policy_indice[player]
            br_policy_per_player.append(self._policies[br_idx])
            br_policy_value_per_player.append(policy_values[player][br_idx])
            br_distribution_per_player.append((self._distributions[br_idx]))
            # br_distribution_per_player.append(self._distributions[br_idx])
            if self._model is None:
                self._policy_frequency_counter[player][br_idx] += 1


        regret = self.regret(sum(br_policy_value_per_player))

        ### Test information
        # print("policy_values:", policy_values)
        # print("br_policy_per_player:", br_policy_per_player)
        # print("br_policy_value_per_player:", br_policy_value_per_player)
        # print("br_distribution_per_player:", br_distribution_per_player)
        # print("regret:", regret)
        # print("***************************")

        if self._model is not None:
            self._updated_policy = ILMergedPolicy(game=self._mfg_game,
                                                  player_ids=list(range(self._num_players)),
                                                  prev_policy=self._updated_policy,
                                                  prev_distribution=self._updated_policy_distribution,
                                                  br_policies=model_br_policy_per_player,
                                                  br_distributions=model_br_distribution_per_player,
                                                  weights=self.update_weights()).to_tabular()

        else:
            self._updated_policy = ILMergedPolicy(game=self._mfg_game,
                                                  player_ids=list(range(self._num_players)),
                                                  prev_policy=self._updated_policy,
                                                  prev_distribution=self._updated_policy_distribution,
                                                  br_policies=br_policy_per_player,
                                                  br_distributions=br_distribution_per_player,
                                                  weights=self.update_weights()).to_tabular()

        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        return regret

    def run(self):
        """
        Run fictitious play.
        """
        regret_list = []
        for _ in range(self._num_iterations):
            regret = self.iteration()
            regret_list.append(regret)
            self._current_iter += 1
            if regret < self._regret_threshold:
                break
        print("FP regret:", regret_list)

    def predict(self, policy_idx):
        """ Using model to predict the utility given s and mu."""
        if self._model is None:
            raise ValueError("The model has not been specified.")
        one_hot_policy = np.zeros(len(self._policies))
        one_hot_policy[policy_idx] = 1

        weights = self.get_weights_on_orig_policies()[0]

        if self._norm:
            norm_weights = (weights - np.mean(weights)) / np.std(weights)
            X = np.append(one_hot_policy, norm_weights)
        else:
            X = np.append(one_hot_policy, weights)

        X = X[None,...]
        estimate_Y = np.squeeze(self._model.predict(X))[()]

        return estimate_Y

    def regret(self, deviation_payoff):
        """
        Calculate the regret of the self._updated_policy within the empirical game.
        :param deviation_payoff: A sum of players' deviation payoff.
        :return:
        """
        pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, self._updated_policy)
        # print("Payoff:", sum([pol_value.value(state) for state in self._root_states]))
        regret = deviation_payoff - sum([pol_value.value(state) for state in self._root_states])

        return regret

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
Uniform MSS.
"""

class UniformMSS(MetaStrategyMethod):
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions,
                 **kwargs):
        super(UniformMSS, self).__init__(mfg_game, policies, distributions)

    def run(self):
        self._updated_policy = MergedPolicy(
                        self._mfg_game, list(range(self._num_players)),
                        self._policies, self._distributions,
                        [1.0 / len(self._policies) for _ in range(len(self._policies))]
                        ).to_tabular()

    def get_output_policies(self):
        """
        Only valid after run().
        :return: the output policies.
        """
        return self._updated_policy

    def get_weights_on_orig_policies(self):
        """
        Get weights on the self._policies.
        :return:
        """
        return [1.0/len(self._policies) for _ in range(self._policies)]

    def reset(self):
        """ Dummy reset. """
        pass


"""
RD MSS.
"""
class ReplicatorDynamicsMSS(MetaStrategyMethod):
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions,
                 num_iterations,
                 root_state=None,
                 dt=0.03,
                 regret_threshold=0.02,
                 model=None):
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
        self._weights = [np.ones(len(self._policies))/len(self._policies) for _ in range(self._num_players)]

        if self._model is not None:
            self._model_weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
                             range(self._num_players)]


    def reset(self):
        """
        Reset the current policies.
        """
        # Initialize with uniform policy.
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
        # self._weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
        #                                   range(self._num_players)]
        #
        # if self._model is not None:
        #     self._model_weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
        #                      range(self._num_players)]

        # Start RD with equilibrium strategy from previous iteration.
        self._current_iter = 0
        if len(self._policies) > 1:  # No zero to add at the first iter.
            for player in range(self._num_players):
                self._weights[player] = np.pad(self._weights[player], (0, 1), 'constant')
                self._weights[player][-1] = 0.005
                self._weights[player][0] -= 0.005


    def update_weights(self, weights, values):
        """ Update RD weights. """
        for player in range(self._num_players):
            weights[player] += weights[player] * (np.array(values[player]) - np.sum(weights[player] * np.array(values[player]))) * self._dt
            if np.sum(weights[player]) != 1:
                weights[player] /= np.sum(weights[player])


    def iteration(self):
        # This aims at fine-tuning RD.
        # if self._model is not None and self._current_iter == 19:
        #     self._model = None
        #     self._weights = self._model_weights

        policy_values = [[] for _ in range(self._num_players)]
        if self._model is not None:
            model_policy_values = [[] for _ in range(self._num_players)]

        for idx, policy in enumerate(self._policies):
            if self._model is not None:
                model_pol_value = self.predict(idx)
            pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, policy)
            for player, state in enumerate(self._root_states):
                if self._model is not None:
                    model_policy_values[player].append(model_pol_value)
                policy_values[player].append(pol_value.value(state))


        regret = self.regret(sum(np.max(policy_values, axis=1)))

        if self._model is not None:
            self.update_weights(self._model_weights, model_policy_values)
        else:
            self.update_weights(self._weights, policy_values)

        if self._model is not None:
            self._updated_policy = MergedPolicy(
                self._mfg_game, list(range(self._num_players)),
                self._policies, self._distributions,
                self._model_weights[0]
            ).to_tabular()
        else:
            self._updated_policy = MergedPolicy(
                self._mfg_game, list(range(self._num_players)),
                self._policies, self._distributions,
                self._weights[0] # ONLY WORK FOR SINGLE POPULATION.
            ).to_tabular()
        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        return regret

    def run(self):
        """
        Run Replicator Dynamics.
        """
        regret_list = []
        for _ in range(self._num_iterations):
            regret = self.iteration()
            regret_list.append(regret)
            self._current_iter += 1
            if regret < self._regret_threshold:
                break
        print("RD regret:", regret_list)


    def regret(self, deviation_payoff):
        """
        Calculate the regret of the self._updated_policy within the empirical game.
        :param deviation_payoff: A sum of players' deviation payoff.
        """
        pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, self._updated_policy)
        # print("Payoff:", sum([pol_value.value(state) for state in self._root_states]))
        regret = deviation_payoff - sum([pol_value.value(state) for state in self._root_states])

        return regret


    def predict(self, policy_idx):
        """ Using model to predict the utility given s and mu."""
        if self._model is None:
            raise ValueError("The model has not been specified.")
        one_hot_policy = np.zeros(len(self._policies))
        one_hot_policy[policy_idx] = 1

        weights = self.get_model_weights()[0]

        X = np.append(one_hot_policy, weights)

        X = X[None,...]
        estimate_Y = np.squeeze(self._model.predict(X))[()]

        return estimate_Y


    def get_weights(self):
        return self._weights


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



class IteratedQuantalResponse(MetaStrategyMethod):
    def __init__(self,
                 mfg_game,
                 policies,
                 distributions,
                 num_iterations,
                 root_state=None,
                 lam=1,
                 regret_threshold=0.02,
                 model=None):
        """
        Apply Iterated quantal best response to the empirical game.
        ONLY WORK FOR SINGLE POPULATION. (Different players should have different weights).
        :param mfg_game: The full game to analyze.
        :param policies: A `List[policy_std.Policy]` object.
        :param distributions: A `List[distribution_std.Distribution]` object corresponding to the policies.
        :param num_iterations: Number of inner loop iterations.
        """
        super(IteratedQuantalResponse, self).__init__(mfg_game, policies, distributions)
        self._num_iterations = num_iterations
        self._regret_threshold = regret_threshold
        self._lambda = lam

        if self._num_players > 1:
            raise NotImplementedError("Currently only work for single population.")

        #Model-based Learning
        self._model = model

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
        self._weights = [np.ones(len(self._policies))/len(self._policies) for _ in range(self._num_players)]

        if self._model is not None:
            self._model_weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
                             range(self._num_players)]


    def reset(self):
        """
        Reset the current policies.
        """
        # Initialize with uniform policy.
        self._updated_policy = MergedPolicy(
            self._mfg_game, list(range(self._num_players)),
            self._policies, self._distributions,
            [1.0 / len(self._policies) for _ in range(len(self._policies))]
        ).to_tabular()
        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        # Current number of iterations.
        self._current_iter = 0

        # The frequency of each policy being a best response in the empirical game, one frequency array per player.
        self._weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
                                          range(self._num_players)]

        if self._model is not None:
            self._model_weights = [np.ones(len(self._policies)) / len(self._policies) for _ in
                             range(self._num_players)]


    def update_weights(self, weights, values):
        """ Update RD weights. """
        for player in range(self._num_players):
            weights[player] = np.exp(self._lambda * np.array(values[player])) / np.sum(np.exp(self._lambda * np.array(values[player])))
            if np.sum(weights[player]) != 1:
                weights[player] /= np.sum(weights[player])


    def iteration(self):
        policy_values = [[] for _ in range(self._num_players)]
        if self._model is not None:
            model_policy_values = [[] for _ in range(self._num_players)]

        for idx, policy in enumerate(self._policies):
            if self._model is not None:
                model_pol_value = self.predict(idx)
            pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, policy)
            for player, state in enumerate(self._root_states):
                if self._model is not None:
                    model_policy_values[player].append(model_pol_value)
                policy_values[player].append(pol_value.value(state))


        regret = self.regret(sum(np.max(policy_values, axis=1)))

        if self._model is not None:
            self.update_weights(self._model_weights, model_policy_values)
        else:
            self.update_weights(self._weights, policy_values)

        if self._model is not None:
            self._updated_policy = MergedPolicy(
                self._mfg_game, list(range(self._num_players)),
                self._policies, self._distributions,
                self._model_weights[0]
            ).to_tabular()
        else:
            self._updated_policy = MergedPolicy(
                self._mfg_game, list(range(self._num_players)),
                self._policies, self._distributions,
                self._weights[0] # ONLY WORK FOR SINGLE POPULATION.
            ).to_tabular()
        self._updated_policy_distribution = distribution.DistributionPolicy(self._mfg_game, self._updated_policy)

        return regret

    def run(self):
        """
        Run Replicator Dynamics.
        """
        regret_list = []
        for _ in range(self._num_iterations):
            regret = self.iteration()
            regret_list.append(regret)
            self._current_iter += 1
            if regret < self._regret_threshold:
                break
        print("QRB regret:", regret_list)


    def regret(self, deviation_payoff):
        """
        Calculate the regret of the self._updated_policy within the empirical game.
        :param deviation_payoff: A sum of players' deviation payoff.
        """
        pol_value = policy_value.PolicyValue(self._mfg_game, self._updated_policy_distribution, self._updated_policy)
        # print("Payoff:", sum([pol_value.value(state) for state in self._root_states]))
        regret = deviation_payoff - sum([pol_value.value(state) for state in self._root_states])

        return regret


    def predict(self, policy_idx):
        """ Using model to predict the utility given s and mu."""
        if self._model is None:
            raise ValueError("The model has not been specified.")
        one_hot_policy = np.zeros(len(self._policies))
        one_hot_policy[policy_idx] = 1

        weights = self.get_model_weights()[0]

        X = np.append(one_hot_policy, weights)

        X = X[None,...]
        estimate_Y = np.squeeze(self._model.predict(X))[()]

        return estimate_Y


    def get_weights(self):
        return self._weights


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
    "nash": FictitiousPlayMSS,
    "uniform": UniformMSS,
    "RD": ReplicatorDynamicsMSS,
    "QRB": IteratedQuantalResponse
}