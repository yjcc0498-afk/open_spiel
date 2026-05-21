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


""" Example of running EGTA on mean field games. """

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
from open_spiel.games.mfg.EGTA import benchmark
from open_spiel.python.mfg.games import predator_prey
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.games.mfg.EGTA.utils import list_to_txt
import pyspiel

FLAGS = flags.FLAGS

# EGTA configurations
flags.DEFINE_string("game_name", "mfg_crowd_modelling", " Mean field game names 'python_mfg_crowd_modelling','mfg_crowd_modelling','mfg_crowd_modelling_2d', 'python_mfg_predator_prey'.")
flags.DEFINE_string("oracle_type", "BR", "Choice is BR (exact Best Response), others NotImplemented")
flags.DEFINE_integer("egta_iterations", 10,
                     "Number of EGTA iterations.")
flags.DEFINE_integer("IL_iterations", 10,
                     "Number of inner loop iterations.")
flags.DEFINE_string("meta_strategy_method", "nash",
                    "Name of meta strategy computation method.")
flags.DEFINE_float("regret_threshold", 0.02, "Regret for early stopping.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")

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
                                      meta_strategy_method=FLAGS.meta_strategy_method,
                                      regret_threshold=FLAGS.regret_threshold)

    start_time = time.time()
    egta_exp = []
    egta_exp_consistent = []
    for egta_iteration in range(FLAGS.egta_iterations):
        if FLAGS.verbose:
            print("\n===========================\n")
            print("Iteration : {}".format(egta_iteration))
            print("Time so far: {}".format(time.time() - start_time))
        consistent_policy = egta_solver.iteration()
        _, meta_probabilities = egta_solver.get_original_policies_and_weights()
        policy = egta_solver.get_merged_policy()

        if FLAGS.verbose:
            print("Probabilities : {}".format(meta_probabilities))
            nashconv = nash_conv.NashConv(game, policy)
            nashconv_value = nashconv.nash_conv()

            # Consistency
            consistent_nashconv = nash_conv.NashConv(game, consistent_policy)
            consistent_nashconv_value = consistent_nashconv.nash_conv()

            writer.add_scalar('egta_exp', nashconv_value, egta_iteration)
            writer.add_scalar('egta_exp_consistent', consistent_nashconv_value, egta_iteration)
            egta_exp.append(nashconv_value)
            egta_exp_consistent.append(consistent_nashconv_value)
            print("NashConv : {}".format(nashconv_value))
            print("Consistent NashConv : {}".format(consistent_nashconv_value))

    list_to_txt(checkpoint_dir + '/egta_exp.txt', egta_exp)
    list_to_txt(checkpoint_dir + '/egta_exp_consistent.txt', egta_exp_consistent)

    # fp = fictitious_play.FictitiousPlay(game)
    # print("================================")
    # print("Beginning running FP benchmark.")
    # for j in range(FLAGS.egta_iterations):
    #     fp.iteration()
    #     fp_policy = fp.get_policy()
    #     nash_conv_fp = nash_conv.NashConv(game, fp_policy)
    #     nashconv_value = nash_conv_fp.nash_conv()
    #     writer.add_scalar('fp_exp', nashconv_value, j)
    #     print("Iteration : {}".format(j), "NashConv : {}".format(nashconv_value))
    #
    #
    # md = mirror_descent.MirrorDescent(game)
    # print("================================")
    # print("Beginning running MD benchmark.")
    # for j in range(FLAGS.egta_iterations):
    #     md.iteration()
    #     md_policy = md.get_policy()
    #     nash_conv_md = nash_conv.NashConv(game, md_policy)
    #     nashconv_value = nash_conv_md.nash_conv()
    #     writer.add_scalar('md_exp', nashconv_value, j)
    #     print("Iteration : {}".format(j), "NashConv : {}".format(nashconv_value))



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Handle standard output.
    seed = np.random.randint(low=0, high=1e5)
    checkpoint_dir = FLAGS.game_name
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)
    checkpoint_dir += '_it_' + str(FLAGS.egta_iterations) + '_or_' + FLAGS.oracle_type + '_heur_' + FLAGS.meta_strategy_method + '_reg_' + str(FLAGS.regret_threshold) + '_se_' + str(seed)+'_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    writer = SummaryWriter(logdir=checkpoint_dir + '/log')
    sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')


    # Main controller.
    if FLAGS.game_name == "python_mfg_predator_prey":
        game = predator_prey.MFGPredatorPreyGame({"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    else:
        game = pyspiel.load_game(FLAGS.game_name, {"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    egta_looper(game=game, writer=writer, checkpoint_dir=checkpoint_dir)
    # benchmark.fp_benchmark(mfg_game=game, num_iterations=FLAGS.egta_iterations, writer=writer)
    # benchmark.md_benchmark(mfg_game=game, num_iterations=FLAGS.egta_iterations, writer=writer)

if __name__ == "__main__":
    app.run(main)

















