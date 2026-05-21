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
from open_spiel.python.mfg.algorithms.EGTA import init_oracle
import pyspiel


class InitOracleTest(absltest.TestCase):

    def test_init_br_oracle(self):
        game = crowd_modelling.MFGCrowdModellingGame()
        oralce, init_policies, init_dist = init_oracle.init_br_oracle(game, None)


if __name__ == "__main__":
  absltest.main()
