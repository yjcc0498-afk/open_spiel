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

"Sample data from a restricted game."

from absl import app
from absl import flags
import time
import datetime
import os
import sys
import functools
print = functools.partial(print, flush=True)

import numpy as np
from tensorboardX import SummaryWriter
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.games.mfg.EGTA import egta
from open_spiel.games.mfg.EGTA.model_learning.sample_utility import Coarse_Utility_Sampler, Finer_Utility_Sampler
from open_spiel.games.mfg.EGTA.game_presets import apply_table4_preset
from open_spiel.python.mfg.games import predator_prey
import pyspiel



FLAGS = flags.FLAGS

# EGTA configurations
flags.DEFINE_string("game_name", "mfg_crowd_modelling", " Mean field game names 'python_mfg_crowd_modelling','mfg_crowd_modelling','mfg_crowd_modelling_2d', 'python_mfg_predator_prey', "
                                                        "'mfg_garnet', 'python_mfg_dynamic_routing', 'mean_field_lin_quad'.")
flags.DEFINE_string("oracle_type", "BR", "Choice is BR (exact Best Response), others NotImplemented")
flags.DEFINE_integer("egta_iterations", 10,
                     "Number of EGTA iterations.")
flags.DEFINE_integer("IL_iterations", 10,
                     "Number of inner loop iterations.")
flags.DEFINE_integer("num_samples", 500,
                     "Number of inner loop iterations.")
flags.DEFINE_string("meta_strategy_method", "nash",
                    "Name of meta strategy computation method.")
flags.DEFINE_bool("grid", True, "Enables verbose printing and profiling.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
flags.DEFINE_bool("finer_encode", False, "Enables finer encoding.")
# 新增：显式控制 utility 数据用哪种策略表示。
# one_hot 保留旧版基线；transformer_stats 生成新方案需要的时序特征数据。
flags.DEFINE_string("encoding", "one_hot", "Utility encoding: 'one_hot' or 'transformer_stats'.")
flags.DEFINE_enum("sampling_mode", "hybrid", ["grid", "dirichlet", "hybrid"],
                  "Mixed-strategy sampling scheme for paper reproduction.")
flags.DEFINE_integer("grid_density", 4, "Simplex grid density for grid samples.")
flags.DEFINE_integer("grid_sample_count", 0, "Grid samples to keep. 0 splits num_samples automatically.")
flags.DEFINE_integer("dirichlet_sample_count", 0, "Dirichlet samples to draw. 0 splits num_samples automatically.")

# Game configuration
flags.DEFINE_integer("game_size", 10,
                     "Game size.")
flags.DEFINE_integer("game_horizon", 10,
                     "Game horizon.")

# System configurations
flags.DEFINE_string("root_result_folder",'root_result',"root directory of saved results")


def egta_looper(game, writer, checkpoint_dir):
    """Initializes and executes the EGTA training loop for mean field games."""

    egta_solver = egta.MFGMetaTrainer(mfg_game=game,
                                      oracle_type=FLAGS.oracle_type,
                                      num_inner_iters=FLAGS.IL_iterations,
                                      meta_strategy_method=FLAGS.meta_strategy_method)


    for egta_iteration in range(FLAGS.egta_iterations):
        if FLAGS.verbose:
            print("\n===========================\n")
            print("Iteration : {}".format(egta_iteration))
        egta_solver.iteration()
        _, meta_probabilities = egta_solver.get_original_policies_and_weights()
        policy = egta_solver.get_merged_policy()

        if FLAGS.verbose:
            print("Probabilities : {}".format(meta_probabilities))
            nashconv = nash_conv.NashConv(game, policy)
            nashconv_value = nashconv.nash_conv()
            writer.add_scalar('egta_exp', nashconv_value, egta_iteration)
            print("NashConv : {}".format(nashconv_value))


    ## Game Model Learning ###
    grid_sample_count = FLAGS.grid_sample_count if FLAGS.grid_sample_count > 0 else None
    dirichlet_sample_count = FLAGS.dirichlet_sample_count if FLAGS.dirichlet_sample_count > 0 else None
    if FLAGS.finer_encode:
        print("Starting generating finer data.")
        finer_sampler = Finer_Utility_Sampler(mfg_game=game,
                                         egta_solver=egta_solver,
                                         num_samples=FLAGS.num_samples,
                                         size=FLAGS.game_size,
                                         horizon=FLAGS.game_horizon,
                                         checkpoint_dir=checkpoint_dir,
                                         grid=FLAGS.grid,
                                         grid_density=FLAGS.grid_density)

        finer_sampler.compute_utility()
    else:
        print("Starting generating coarse data.")
        coarse_sampler = Coarse_Utility_Sampler(mfg_game=game,
                                         egta_solver=egta_solver,
                                         num_samples=FLAGS.num_samples,
                                         checkpoint_dir=checkpoint_dir,
                                         grid=FLAGS.grid,
                                         grid_density=FLAGS.grid_density,
                                         encoding=FLAGS.encoding,
                                         sampling_mode=FLAGS.sampling_mode,
                                         grid_sample_count=grid_sample_count,
                                         dirichlet_sample_count=dirichlet_sample_count)

        coarse_sampler.compute_utility()



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # 新增：对论文 Table 4 中列出的游戏自动应用预设参数，
    # 让不同 game 默认遵从 states / horizon / strategies / samples 的配置表。
    apply_table4_preset(FLAGS.game_name, FLAGS)

    # Handle standard output.
    seed = np.random.randint(low=0, high=1e5)
    checkpoint_dir = FLAGS.game_name
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)
    checkpoint_dir += '_it_' + str(FLAGS.egta_iterations) + '_sampling_' + FLAGS.sampling_mode + '_sample_' + str(FLAGS.num_samples) + '_gendata_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    writer = SummaryWriter(logdir=checkpoint_dir + '/log')
    sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')


    # Main controller.
    if FLAGS.game_name == "python_mfg_predator_prey":
        game = predator_prey.MFGPredatorPreyGame({"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    elif FLAGS.game_name == 'mean_field_lin_quad':
        game = pyspiel.load_game(FLAGS.game_name, {"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    else:
        game = pyspiel.load_game(FLAGS.game_name, {"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    egta_looper(game=game, writer=writer, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    app.run(main)
