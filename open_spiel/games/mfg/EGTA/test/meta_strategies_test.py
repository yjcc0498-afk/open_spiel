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

from absl.testing import absltest
import time
import numpy as np

from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling, predator_prey
from open_spiel.python.mfg.algorithms.EGTA.meta_strategies import FictitiousPlayMSS
import pyspiel

class MSSTest(absltest.TestCase):

    def test_fp_single_population(self):
        """ Test fictitious player inner loop for single-population game."""
        # game = crowd_modelling.MFGCrowdModellingGame()
        game = pyspiel.load_game("mfg_crowd_modelling", {"size": 10})

        perturbed_policy = policy.UniformRandomPolicy(game).to_tabular()
        perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)
        policies = [perturbed_policy]
        distributions = [perturbed_distribution]
        for _ in range(5):
            perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)

            br_value = best_response_value.BestResponse(game, perturbed_distribution)
            greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
            greedy_pi = greedy_pi.to_tabular()
            distrib_greedy = distribution.DistributionPolicy(game, greedy_pi)

            policies.append(greedy_pi)
            distributions.append(distrib_greedy)
            perturbed_policy = greedy_pi

        fp_meta_strategy_method = FictitiousPlayMSS(mfg_game=game,
                                                    policies=policies,
                                                    distributions=distributions,
                                                    num_iterations=100)

        fp_meta_strategy_method.run()
        updated_policy = fp_meta_strategy_method.get_output_policies()
        weights = fp_meta_strategy_method.get_weights_on_orig_policies()
        print("Final weights:", weights, fp_meta_strategy_method.get_current_iter())

    # def test_merged_vs_expected_values(self):
    #     """Merge policy value should be different from weighted sum of individual policy value."""
    #     game = crowd_modelling.MFGCrowdModellingGame()
    #
    #     perturbed_policy = policy.UniformRandomPolicy(game).to_tabular()
    #     perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)
    #     policies = [perturbed_policy]
    #     distributions = [perturbed_distribution]
    #     for _ in range(4):
    #         perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)
    #
    #         br_value = best_response_value.BestResponse(game, perturbed_distribution)
    #         greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    #         greedy_pi = greedy_pi.to_tabular()
    #         distrib_greedy = distribution.DistributionPolicy(game, greedy_pi)
    #
    #         policies.append(greedy_pi)
    #         distributions.append(distrib_greedy)
    #         perturbed_policy = greedy_pi
    #
    #     merged_policy = fictitious_play.MergedPolicy(
    #         game, list(range(game.num_players())),
    #         policies, distributions,
    #         [0.2, 0.2, 0.2, 0.2, 0.2]
    #     ).to_tabular()
    #
    #     merged_distribution = distribution.DistributionPolicy(game, merged_policy)
    #
    #     merged_pol_value = policy_value.PolicyValue(game, merged_distribution, merged_policy)
    #
    #     # pol_values = [merged_pol_value]
    #     # for p, d in zip(policies, distributions):
    #     #     pol_values.append(policy_value.PolicyValue(game, d, p))
    #
    #     pol_values = [merged_pol_value]
    #     for p in policies:
    #         pol_values.append(policy_value.PolicyValue(game, merged_distribution, p))
    #
    #     v_values = []
    #     for v in pol_values:
    #         for state in game.new_initial_states():
    #             v_values.append(v.eval_state(state))
    #
    #     print(v_values[0], np.sum(np.array(v_values[1:]) * np.array([0.2, 0.2, 0.2, 0.2, 0.2])))
        # self.assertAlmostEqual(v_values[0], np.sum(np.array(v_values[1:]) * np.array([0.2, 0.2, 0.2, 0.2, 0.2])))


    # def test_policy_value_single_population(self):
    #     """ Test different policy values in CrowdModeling. """
    #     game = crowd_modelling.MFGCrowdModellingGame()
    #
    #     uniform_policy = policy.UniformRandomPolicy(game)
    #     uniform_policy_distribution = distribution.DistributionPolicy(game, uniform_policy)
    #
    #     FA_policy = policy.FirstActionPolicy(game)
    #     FA_policy_distribution = distribution.DistributionPolicy(game, FA_policy)
    #
    #     br_value = best_response_value.BestResponse(game, uniform_policy_distribution)
    #     greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    #     greedy_pi = greedy_pi.to_tabular()
    #     distrib_greedy = distribution.DistributionPolicy(game, greedy_pi)
    #
    #     unif_pol_value = policy_value.PolicyValue(game, uniform_policy_distribution, uniform_policy)
    #     FA_pol_value = policy_value.PolicyValue(game, FA_policy_distribution, FA_policy)
    #     greedy_pol_value = policy_value.PolicyValue(game, distrib_greedy, greedy_pi)
    #
    #     # for state in game.new_initial_states():
    #     #     print("unif_pol_value:", unif_pol_value.eval_state(state))
    #     #     print("FA_pol_value:", FA_pol_value.eval_state(state))
    #     #     print("greedy_pol_value:", greedy_pol_value(state))


    # def test_fp_multi_populations(self):
    #     """ Test fictitious player inner loop for Multi-population game."""
    #     # game = predator_prey.MFGPredatorPreyGame()
    #     game = pyspiel.load_game("python_mfg_predator_prey")
    #
    #     perturbed_policy = policy.UniformRandomPolicy(game).to_tabular()
    #     perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)
    #     policies = [perturbed_policy]
    #     distributions = [perturbed_distribution]
    #
    #     t1 = time.time()
    #     for _ in range(4):
    #         perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)
    #
    #         br_value = best_response_value.BestResponse(game, perturbed_distribution)
    #         greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    #         greedy_pi = greedy_pi.to_tabular()
    #         distrib_greedy = distribution.DistributionPolicy(game, greedy_pi)
    #
    #         policies.append(greedy_pi)
    #         distributions.append(distrib_greedy)
    #         perturbed_policy = greedy_pi
    #
    #     t2 = time.time()
    #     print("BEGIN FP METHOD.....", t2 - t1)
    #     fp_meta_strategy_method = FictitiousPlayMSS(mfg_game=game,
    #                                                 policies=policies,
    #                                                 distributions=distributions,
    #                                                 num_iterations=5)
    #
    #     fp_meta_strategy_method.run()
    #     updated_policy = fp_meta_strategy_method.get_output_policies()
    #     weights = fp_meta_strategy_method.get_weights_on_orig_policies()
    #     print("Final weights:", weights)




if __name__ == "__main__":
  absltest.main()