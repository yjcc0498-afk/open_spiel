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


from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python import rl_environment


def init_br_oracle(game, initial_policy):
    """Initializes the tabular best-response based responder and agents."""
    if initial_policy:
        init_pol = initial_policy
    else:
        init_pol = policy_std.UniformRandomPolicy(game).to_tabular()
    init_distribution = distribution.DistributionPolicy(game, init_pol)

    # Set up tabular oracle.
    def oracle(game, distribution):
        """
        Best response oracle to a distribution.
        :param game: a MFG.
        :param distribution: A merged distribution over players.
        :return: A best-response policy.
        """
        br_value = best_response_value.BestResponse(game, distribution)
        greedy_pi = greedy_policy.GreedyPolicy(game, list(range(game.num_players())), br_value)
        greedy_pi = greedy_pi.to_tabular()
        return greedy_pi

    return oracle, [init_pol], [init_distribution]




def init_dqn_oracle(game, initial_policies):
    pass
