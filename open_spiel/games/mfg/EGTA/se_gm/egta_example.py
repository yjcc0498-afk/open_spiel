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
from open_spiel.python.mfg.algorithms.EGTA import egta
from open_spiel.python.mfg.algorithms.EGTA.se_gm.model import TF_Regressor
from open_spiel.games.mfg.EGTA.game_presets import apply_table4_preset
from open_spiel.python.mfg.games import predator_prey
from open_spiel.python.mfg.algorithms.EGTA.utils import list_to_txt
import pyspiel

FLAGS = flags.FLAGS

# EGTA configurations
flags.DEFINE_string("game_name", "mfg_crowd_modelling", " Mean field game names 'python_mfg_crowd_modelling','mfg_crowd_modelling','mfg_crowd_modelling_2d', 'python_mfg_predator_prey'.")
flags.DEFINE_string("oracle_type", "BR", "Choice is BR (exact Best Response), others NotImplemented")
flags.DEFINE_integer("egta_iterations", 10,
                     "Number of EGTA iterations.")
flags.DEFINE_integer("IL_iterations", 10,
                     "Number of inner loop iterations.")
flags.DEFINE_string("meta_strategy_method", "RD",
                    "Name of meta strategy computation method.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
# 新增：训练/评估入口允许在旧 one-hot 基线和新 Transformer 编码之间切换。
flags.DEFINE_string("encoding", "one_hot", "Model input encoding: 'one_hot' or 'transformer_stats'.")

flags.DEFINE_integer("planning_iters", 5,
                     "Num of iterations for FP/RD with Models.")
flags.DEFINE_integer("fine_tune_iters", 5,
                     "Num of samples for FP/RD with true utility function.")
flags.DEFINE_float("w_distance", 0.015, "Threshold for Wasserstein distance to measure quality of the model.")

# Game configuration
flags.DEFINE_integer("game_size", 10,
                     "Game size.")
flags.DEFINE_integer("game_horizon", 10,
                     "Game horizon.")

# System configurations
flags.DEFINE_string("root_result_folder", 'root_result', "root directory of saved results")



# Hard-coded hyperparams.
HP = {}
HP['learning_rate'] = 0.001
HP['epoch'] = 500
HP['batch_size'] = 32
HP['output_sizes'] = [256, 256, 1]
HP['feature_dim'] = 6
HP['d_model'] = 128
HP['nhead'] = 4
HP['num_layers'] = 2


# Testing NN hyperparams.
# HP = {}
# HP['learning_rate'] = 0.001
# HP['epoch'] = 10
# HP['batch_size'] = 32
# HP['output_sizes'] = [64, 64, 1]



def egta_looper(game, writer, checkpoint_dir):
    """Initializes and executes the EGTA training loop for mean field games."""

    # Initialize the model (i.e., a utility simulator).
    # 新增：把 Transformer 所需的超参数和编码模式一起传给 regressor，
    # 这样 outer loop 不需要知道模型内部是 MLP 还是 Transformer。
    model_hp = dict(HP)
    model_hp['num_policies'] = FLAGS.egta_iterations + 1
    model_hp['sequence_length'] = FLAGS.game_horizon
    model = TF_Regressor(
        nn_params=model_hp,
        verbose=0,
        checkpoint_dir=checkpoint_dir,
        encoding=FLAGS.encoding)
    print("NN params:", HP)

    # Initialize EGTA.
    egta_solver = egta.MFGMetaTrainer(mfg_game=game,
                                      model=model,
                                      oracle_type=FLAGS.oracle_type,
                                      num_inner_iters=FLAGS.IL_iterations,
                                      meta_strategy_method=FLAGS.meta_strategy_method,
                                      EGTA_iterations=FLAGS.egta_iterations,
                                      planning_iters=FLAGS.planning_iters,
                                      fine_tune_iters=FLAGS.fine_tune_iters,
                                      w_distance=FLAGS.w_distance)

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


    model.save_model()
    list_to_txt(checkpoint_dir + '/egta_exp.txt', egta_exp)
    list_to_txt(checkpoint_dir + '/egta_exp_consistent.txt', egta_exp_consistent)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # 新增：对论文 Table 4 中的三个目标游戏自动套用预设，
    # 让 game size / horizon / 策略数与论文超参数表保持一致。
    apply_table4_preset(FLAGS.game_name, FLAGS)

    # Handle standard output.
    seed = np.random.randint(low=0, high=1e5)
    checkpoint_dir = FLAGS.game_name
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)
    checkpoint_dir += '_size_' + str(FLAGS.game_size) + '_T_' + str(FLAGS.game_horizon) + '_it_' + str(FLAGS.egta_iterations) + '_pl_' + str(FLAGS.planning_iters) + '_fi_' + str(FLAGS.fine_tune_iters)\
                      + '_heur_' + FLAGS.meta_strategy_method + "_dist_" + str(FLAGS.w_distance) + '_IL_' + str(FLAGS.IL_iterations) + '_se_'+str(seed)+'_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    writer = SummaryWriter(logdir=checkpoint_dir + '/log')
    sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')


    # Main controller.
    if FLAGS.game_name == "python_mfg_predator_prey":
        game = predator_prey.MFGPredatorPreyGame({"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    else:
        game = pyspiel.load_game(FLAGS.game_name, {"size": FLAGS.game_size, "horizon": FLAGS.game_horizon})
    egta_looper(game=game, writer=writer, checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":
    app.run(main)














