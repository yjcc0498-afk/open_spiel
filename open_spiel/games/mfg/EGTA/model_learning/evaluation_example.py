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
import datetime
import os
import sys
import functools
print = functools.partial(print, flush=True)

import numpy as np
from tensorboardX import SummaryWriter
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms.EGTA import egta, meta_strategies
from open_spiel.python.mfg.games import predator_prey
from open_spiel.python.mfg.algorithms.EGTA.model_learning import finer_format_data
from open_spiel.python.mfg.algorithms import distribution
import pyspiel


from tensorflow.keras.models import load_model


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

    # Use learned model for the last iteration.
    print("==========Start Running Evaluation========")

    if FLAGS.game_name == "mfg_crowd_modelling":
        model = load_model('./models/tf_model_crowd1d.h5')
        two_dim = False
    elif FLAGS.game_name == "mfg_crowd_modelling_2d":
        model = load_model('./models/tf_model_crowd2d.h5')
        two_dim = True
    elif FLAGS.game_name == "mean_field_lin_quad":
        model = load_model('./models/tf_model_linq.h5')
    else:
        raise ValueError("No specified MFG.")

    formattor = finer_format_data.Formattor(mfg_game=game, size=FLAGS.game_size, horizon=FLAGS.game_horizon, two_dim=two_dim)

    no_model_FP = meta_strategies.MFG_META_STRATEGY_METHODS['nash'](mfg_game=game,
                                                                 policies=egta_solver._policies,
                                                                 distributions=egta_solver._distributions,
                                                                 num_iterations=egta_solver._num_inner_iters)
    no_model_FP.run()

    updated_policy = no_model_FP.get_output_policies()
    updated_policy_distribution = distribution.DistributionPolicy(game, updated_policy)
    dist = formattor.format_distribution(updated_policy_distribution)
    np.save(checkpoint_dir + "/distribution_true.npy", np.array(dist))

    print("==========Start Running Model RD ========")

    model_FP = meta_strategies.MFG_META_STRATEGY_METHODS['nash'](mfg_game=game,
                                                               policies=egta_solver._policies,
                                                               distributions=egta_solver._distributions,
                                                               num_iterations=egta_solver._num_inner_iters,
                                                               model=model)
    model_FP.run()

    updated_policy = model_FP.get_output_policies()
    updated_policy_distribution = distribution.DistributionPolicy(game, updated_policy)
    dist = formattor.format_distribution(updated_policy_distribution)
    np.save(checkpoint_dir + "/distribution_model.npy", np.array(dist))



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Handle standard output.
    seed = np.random.randint(low=0, high=1e5)
    checkpoint_dir = FLAGS.game_name
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)
    checkpoint_dir += '_it_' + str(FLAGS.egta_iterations) + '_eval_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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