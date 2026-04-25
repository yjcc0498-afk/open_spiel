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

from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling, predator_prey
from open_spiel.python.mfg.algorithms.EGTA.meta_strategies import FictitiousPlayMSS
from open_spiel.python.mfg.algorithms.EGTA import inner_loop
import pyspiel

class InnerLoopTest(absltest.TestCase):

    def test_inner_loop(self):
        game = crowd_modelling.MFGCrowdModellingGame()

        perturbed_policy = policy.UniformRandomPolicy(game).to_tabular()
        perturbed_distribution = distribution.DistributionPolicy(game, perturbed_policy)
        policies = [perturbed_policy]
        distributions = [perturbed_distribution]
        for _ in range(4):
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
                                                    num_iterations=5)

        _inner_loop_ = inner_loop.InnerLoop(fp_meta_strategy_method)
        output_merged_policies = _inner_loop_.run_inner_loop()
        print(output_merged_policies)


if __name__ == "__main__":
  absltest.main()