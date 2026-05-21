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

""" The trainer of EGTA for mean field game."""

from open_spiel.python.mfg.algorithms import distribution
from open_spiel.games.mfg.EGTA import meta_strategies
from open_spiel.games.mfg.EGTA import inner_loop
from open_spiel.games.mfg.EGTA import init_oracle

from open_spiel.games.mfg.EGTA.se_gm import meta_strategies as model_meta_strategies

class MFGMetaTrainer(object):
    """
    Empirical game-theoretic analysis (EGTA) for MFGs.
    """
    def __init__(self,
                 mfg_game,
                 oracle_type,
                 num_inner_iters=None,
                 initial_policy=None,
                 meta_strategy_method="nash",
                 regret_threshold=0.02,
                 model=None,
                 planning_iters=0,
                 fine_tune_iters=0,
                 EGTA_iterations=17,
                 w_distance=None,
                 **kwargs):
        """
        Initialize the MFG Trainer.

        :param mfg_game: a mean-field game.
        :param oracle_type: "BR" exact best response or "DQN" RL approximate best response.
        :param num_inner_iters: the number of iterations for the inner loop (finding BR target based on the empirical game) if needed.
        :param initial_policies: initial policies. Uniform policies by default.
        :param meta_strategy_method: method for the inner loop.
        """

        self._mfg_game = mfg_game
        self._oracle_type = oracle_type
        self._num_players = mfg_game.num_players()
        self._num_inner_iters = num_inner_iters
        self._initial_policy = initial_policy
        self._meta_strategy_method_name = meta_strategy_method
        self._EGTA_iterations = EGTA_iterations
        self._regret_threshold = regret_threshold

        #Game model parameters
        self._model = model
        self._planning_iters = planning_iters
        self._fine_tune_iters = fine_tune_iters
        self._w_distance = w_distance

        self.initialize_policies_and_distributions()

        if model is not None:
            self._meta_strategy_method = model_meta_strategies.MFG_META_STRATEGY_METHODS[meta_strategy_method](mfg_game=mfg_game,
                                                                                                               policies=self._policies,
                                                                                                               distributions=self._distributions,
                                                                                                               num_iterations=num_inner_iters,
                                                                                                               EGTA_iterations=EGTA_iterations,
                                                                                                               model=model,
                                                                                                               planning_iters=planning_iters,
                                                                                                               fine_tune_iters=fine_tune_iters,
                                                                                                               w_distance=w_distance)

        else:
            #TODO: check if policy and dist are being updated.
            self._meta_strategy_method = meta_strategies.MFG_META_STRATEGY_METHODS[meta_strategy_method](mfg_game=mfg_game,
                                                                                                         policies=self._policies,
                                                                                                         distributions=self._distributions,
                                                                                                         num_iterations=num_inner_iters,
                                                                                                         regret_threshold=regret_threshold)

        # Add an evaluation solver for consistency.
        self._evaluation_method = meta_strategies.MFG_META_STRATEGY_METHODS['RD'](mfg_game=mfg_game,
                                                                                    policies=self._policies,
                                                                                    distributions=self._distributions,
                                                                                    num_iterations=200)


        self._inner_loop = inner_loop.InnerLoop(self._meta_strategy_method)
        self._output_policy = None

        self._current_outer_iter = 0


    def initialize_policies_and_distributions(self):
        """
        Initialize policies and corresponding distributions.
        """
        if self._oracle_type == "BR":
            self._oracle, self._policies, self._distributions = init_oracle.init_br_oracle(game=self._mfg_game,
                                                                                           initial_policy=self._initial_policy)
        elif self._oracle_type == "DQN":
            raise NotImplementedError
        else:
            raise ValueError("Suggested oracle has not been implemented.")

    def reset(self):
        """
        Reset the trainer.
        """
        self._current_outer_iter = 0
        self.initialize_policies_and_distributions()


        if self._model is not None:
            self._meta_strategy_method = model_meta_strategies.MFG_META_STRATEGY_METHODS[self._meta_strategy_method](mfg_game=self._mfg_game,
                                                                                                               policies=self._policies,
                                                                                                               distributions=self._distributions,
                                                                                                               num_iterations=self._num_inner_iters,
                                                                                                               model=self._model,
                                                                                                               EGTA_iterations=self._EGTA_iterations,
                                                                                                               planning_iters=self._planning_iters,
                                                                                                               fine_tune_iters=self._fine_tune_iters,
                                                                                                               w_distance=self._w_distance)

        else:
            self._meta_strategy_method = meta_strategies.MFG_META_STRATEGY_METHODS[self._meta_strategy_method_name](mfg_game=self._mfg_game,
                                                                                                                 policies=self._policies,
                                                                                                                 distributions=self._distributions,
                                                                                                                 num_iterations=self._num_inner_iters,
                                                                                                                 regret_threshold=self._regret_threshold)

        self._inner_loop = inner_loop.InnerLoop(self._meta_strategy_method)
        self._output_policy = None


    def iteration(self):
        """
        Main training iteration.
        """
        self._current_outer_iter += 1
        self._meta_strategy_method.reset()
        self._output_policy = self._inner_loop.run_inner_loop()
        consistent_policy = self.run_consistency()
        self.update_policies(self._output_policy)

        return consistent_policy


    def final_step(self):
        """ Final analysis of all generated policies. """
        if self._model is None:
            self._meta_strategy_method.reset()
        else:
            self._meta_strategy_method.reset(padding=False)
        self._output_policy = self._inner_loop.run_inner_loop()


    def run_consistency(self):
        """ Run consistency criterion. """
        self._evaluation_method.reset()
        self._evaluation_method.run()
        output_merged_policy = self._evaluation_method.get_output_policies()

        return output_merged_policy


    def update_policies(self, output_merged_policy):
        """
        Adding new best-response policies to the empirical game.
        :param output_merged_policy: a merged policy induced by inner loop.
        :return:
        """
        output_distribution = distribution.DistributionPolicy(self._mfg_game, output_merged_policy)
        greedy_pi = self._oracle(self._mfg_game, output_distribution)

        self._policies.append(greedy_pi)
        self._distributions.append(distribution.DistributionPolicy(self._mfg_game, greedy_pi))

    def update_meta_strategy_solver(self, new_solver):
        """
        Update the
        :param new_solver:
        :return:
        """
        if isinstance(new_solver, str):
            self._meta_strategy_method = meta_strategies.MFG_META_STRATEGY_METHODS[new_solver](mfg_game=self._mfg_game,
                                                                                               policies=self._policies,
                                                                                               distributions=self._distributions,
                                                                                               num_iterations=self._num_inner_iters)

            self._inner_loop = inner_loop.InnerLoop(self._meta_strategy_method)
        else:
            self._meta_strategy_method = new_solver
            self._inner_loop = inner_loop.InnerLoop(new_solver)


    def get_original_policies_and_weights(self):
        """
        Return original policies in the empirical game and corresponding output mixed strategies.
        """
        weights = self._meta_strategy_method.get_weights_on_orig_policies()
        return self._policies, weights

    def get_merged_policy(self):
        """
        Return the output merged policy.
        Equivalent to merge policies and weights from get_original_policies_and_weights().
        """
        return self._output_policy

    def get_inner_loop_stats(self):
        if hasattr(self._meta_strategy_method, "get_last_run_stats"):
            return self._meta_strategy_method.get_last_run_stats()
        return {}

    def get_policies(self):
        return self._policies

    def get_distrbutions(self):
        return self._distributions






