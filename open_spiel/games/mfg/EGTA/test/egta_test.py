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

from open_spiel.python.mfg.games import crowd_modelling
from open_spiel.python.mfg.algorithms.EGTA import egta
import pyspiel

class EGTATest(absltest.TestCase):

    def test_egta(self):
        game = pyspiel.load_game("mfg_crowd_modelling", {"size": 10})
        egta_runner = egta.MFGMetaTrainer(mfg_game=game,
                                          oracle_type="BR",
                                          num_outer_iters=5,
                                          num_inner_iters=5)

        egta_runner.iteration()
        output_policy = egta_runner.get_merged_policy()

if __name__ == "__main__":
  absltest.main()