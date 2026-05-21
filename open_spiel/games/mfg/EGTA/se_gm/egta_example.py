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
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.games.mfg.EGTA import egta
from open_spiel.games.mfg.EGTA.model_learning.finer_format_data import Formattor
from open_spiel.games.mfg.EGTA.se_gm.model import TF_Regressor
from open_spiel.games.mfg.EGTA.game_presets import apply_table4_preset
from open_spiel.python.mfg.games import predator_prey
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
flags.DEFINE_string("meta_strategy_method", "RD",
                    "Name of meta strategy computation method.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
flags.DEFINE_string("encoding", "one_hot", "Model input encoding: 'one_hot' or 'transformer_stats'.")
flags.DEFINE_enum("model_type", "mlp", ["mlp", "transformer", "transformer_stats"],
                  "Utility regressor architecture. transformer keeps the paper one-hot coarse coding input.")
flags.DEFINE_integer("planning_iters", 5,
                     "Num of iterations for FP/RD with Models.")
flags.DEFINE_integer("fine_tune_iters", 5,
                     "Num of samples for FP/RD with true utility function.")
flags.DEFINE_float("w_distance", 0.015, "Threshold for Wasserstein distance to measure quality of the model.")
flags.DEFINE_bool("save_plot_data", True, "Save CSV/NPY files needed for paper-style plots.")

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


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _save_vector(path, values):
    np.savetxt(path, np.asarray(values, dtype=np.float64), delimiter=",")


def _has_values(values):
    return np.asarray(values).size > 0


def _save_inner_loop_stats(checkpoint_dir, egta_iteration, stats):
    if not stats:
        return
    stats_dir = os.path.join(checkpoint_dir, "plot_data", "inner_loop")
    _ensure_dir(stats_dir)
    prefix = os.path.join(stats_dir, "outer_{:03d}".format(egta_iteration))
    for key in ("inner_regret", "model_regret", "model_mask", "norms"):
        values = stats.get(key, [])
        if _has_values(values):
            _save_vector(prefix + "_" + key + ".csv", values)


def _pad_rows(rows):
    max_len = max((len(row) for row in rows), default=0)
    if max_len == 0:
        return np.empty((0, 0), dtype=np.float64)
    output = np.full((len(rows), max_len), np.nan, dtype=np.float64)
    for idx, row in enumerate(rows):
        row = np.asarray(row, dtype=np.float64)
        output[idx, :len(row)] = row
    return output


def _save_stats_summary(checkpoint_dir, all_stats):
    if not all_stats:
        return
    stats_dir = os.path.join(checkpoint_dir, "plot_data")
    _ensure_dir(stats_dir)
    for key in ("inner_regret", "model_regret", "model_mask", "norms"):
        rows = [stats.get(key, []) for stats in all_stats if _has_values(stats.get(key, []))]
        if rows:
            np.savetxt(os.path.join(stats_dir, key + "_by_outer_iteration.csv"),
                       _pad_rows(rows), delimiter=",")
            _save_vector(os.path.join(stats_dir, "final_" + key + ".csv"), rows[-1])


def _format_distribution(game, policy, size, horizon, two_dim):
    policy_distribution = distribution.DistributionPolicy(game, policy)
    formattor = Formattor(mfg_game=game, size=size, horizon=horizon, two_dim=two_dim)
    return np.asarray(formattor.format_distribution(policy_distribution), dtype=np.float64)


def _save_final_distributions(game, checkpoint_dir, learned_policy, true_policy):
    if learned_policy is None or true_policy is None:
        return
    dist_dir = os.path.join(checkpoint_dir, "plot_data", "distributions")
    _ensure_dir(dist_dir)
    two_dim = FLAGS.game_name == "mfg_crowd_modelling_2d"
    learned_dist = _format_distribution(
        game, learned_policy, FLAGS.game_size, FLAGS.game_horizon, two_dim)
    true_dist = _format_distribution(
        game, true_policy, FLAGS.game_size, FLAGS.game_horizon, two_dim)
    np.save(os.path.join(dist_dir, "distribution_model.npy"), learned_dist)
    np.save(os.path.join(dist_dir, "distribution_true.npy"), true_dist)
    if learned_dist.shape == true_dist.shape:
        np.save(os.path.join(dist_dir, "distribution_abs_error.npy"),
                np.abs(learned_dist - true_dist))



def egta_looper(game, writer, checkpoint_dir):
    """Initializes and executes the EGTA training loop for mean field games."""

    # Initialize the model (i.e., a utility simulator).
    # 新增：把 Transformer 所需的超参数和编码模式一起传给 regressor，
    # 这样 outer loop 不需要知道模型内部是 MLP 还是 Transformer。
    model_hp = dict(HP)
    model_hp['num_policies'] = FLAGS.egta_iterations + 1
    model_hp['sequence_length'] = FLAGS.game_horizon
    model_encoding = FLAGS.encoding
    if FLAGS.model_type == "transformer":
        model_encoding = "one_hot_transformer"
    elif FLAGS.model_type == "transformer_stats":
        model_encoding = "transformer_stats"
    model = TF_Regressor(
        nn_params=model_hp,
        verbose=0,
        checkpoint_dir=checkpoint_dir,
        encoding=model_encoding)
    print("NN params:", HP)
    print("Model type:", FLAGS.model_type)
    print("Model encoding:", model_encoding)

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
    all_inner_stats = []
    final_model_policy = None
    final_true_policy = None
    for egta_iteration in range(FLAGS.egta_iterations):
        if FLAGS.verbose:
            print("\n===========================\n")
            print("Iteration : {}".format(egta_iteration))
            print("Time so far: {}".format(time.time() - start_time))

        consistent_policy = egta_solver.iteration()
        _, meta_probabilities = egta_solver.get_original_policies_and_weights()
        policy = egta_solver.get_merged_policy()
        final_model_policy = policy
        final_true_policy = consistent_policy
        if FLAGS.save_plot_data:
            inner_stats = egta_solver.get_inner_loop_stats()
            all_inner_stats.append(inner_stats)
            _save_inner_loop_stats(checkpoint_dir, egta_iteration, inner_stats)

        nashconv = nash_conv.NashConv(game, policy)
        nashconv_value = nashconv.nash_conv()

        # Consistency
        consistent_nashconv = nash_conv.NashConv(game, consistent_policy)
        consistent_nashconv_value = consistent_nashconv.nash_conv()

        writer.add_scalar('egta_exp', nashconv_value, egta_iteration)
        writer.add_scalar('egta_exp_consistent', consistent_nashconv_value, egta_iteration)
        egta_exp.append(nashconv_value)
        egta_exp_consistent.append(consistent_nashconv_value)
        if FLAGS.verbose:
            print("Probabilities : {}".format(meta_probabilities))
            print("NashConv : {}".format(nashconv_value))
            print("Consistent NashConv : {}".format(consistent_nashconv_value))


    model.save_model()
    list_to_txt(checkpoint_dir + '/egta_exp.txt', egta_exp)
    list_to_txt(checkpoint_dir + '/egta_exp_consistent.txt', egta_exp_consistent)
    if FLAGS.save_plot_data:
        _save_stats_summary(checkpoint_dir, all_inner_stats)
        _save_final_distributions(game, checkpoint_dir, final_model_policy, final_true_policy)


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
    checkpoint_dir += '_size_' + str(FLAGS.game_size) + '_T_' + str(FLAGS.game_horizon) + '_it_' + str(FLAGS.egta_iterations) + '_model_' + FLAGS.model_type + '_pl_' + str(FLAGS.planning_iters) + '_fi_' + str(FLAGS.fine_tune_iters)\
                      + '_heur_' + FLAGS.meta_strategy_method + "_dist_" + str(FLAGS.w_distance) + '_IL_' + str(FLAGS.IL_iterations) + '_se_'+str(seed)+'_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)
    _ensure_dir(checkpoint_dir)

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




