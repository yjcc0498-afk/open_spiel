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
import logging

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value


"""
This file has not been tested.
"""

class DQNPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, envs, policies):
    game = envs[0].game
    player_ids = list(range(game.num_players()))
    super(DQNPolicies, self).__init__(game, player_ids)
    self._policies = policies
    self._obs = {
        "info_state": [None] * game.num_players(),
        "legal_actions": [None] * game.num_players()
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (state.observation_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict



class MFGRLOracle(object):
    def __init__(self,
                 mfg_game,
                 best_response_class,
                 best_response_kwargs,
                 number_training_episodes,
                 eval_period,
                 use_checkpoints=False,
                 checkpoint_dir=None):

        self._mfg_game = mfg_game
        self._num_players = mfg_game.num_players
        self._use_checkpoints = use_checkpoints
        self._eval_period = eval_period
        self._checkpoint_dir = checkpoint_dir


        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs

        self._num_train_episodes = number_training_episodes

        uniform_policy = policy.UniformRandomPolicy(mfg_game)
        mfg_dist = distribution.DistributionPolicy(mfg_game, uniform_policy)

        self._envs = [
            rl_environment.Environment(mfg_game, distribution=mfg_dist, mfg_population=p)
            for p in range(self._num_players)
        ]
        self._info_state_size = self._envs[0].observation_spec()["info_state"][0]
        self._num_actions = self._envs[0].action_spec()["num_actions"]

        self._agents = [
            dqn.DQN(idx, self._info_state_size, self._num_actions, **self._best_response_kwargs)
            for idx in range(self._num_players)
        ]
        self._joint_avg_policy = DQNPolicies(self._envs, self._agents)

        # if use_checkpoints:
        #     for agent in agents:
        #         if agent.has_checkpoint(checkpoint_dir):
        #             agent.restore(checkpoint_dir)


    def training(self, distributions):

        for ep in range(self._num_train_episodes):

            # Training monitoring
            if (ep + 1) % self._eval_period == 0:
                losses = [agent.loss for agent in self._agents]
                logging.info("Losses: %s", losses)
                nash_conv_obj = nash_conv.NashConv(self._mfg_game, uniform_policy)
                print(
                    str(ep + 1) + " RL Best Response to Uniform " +
                    str(nash_conv_obj.br_values()))
                pi_value = policy_value.PolicyValue(self._mfg_game, mfg_dist, self._joint_avg_policy)
                print(
                    str(ep + 1) + " DQN Best Response to Uniform " + str([
                        pi_value.eval_state(state)
                        for state in self._mfg_game.new_initial_states()
                    ]))
                if self._use_checkpoints:
                    for agent in self._agents:
                        agent.save(self._checkpoint_dir)
                logging.info("_____________________________________________")


            # Training for one episode.
            for player in range(self._num_players):
                time_step = self._envs[player].reset()
                while not time_step.last():
                    agent_output = self._agents[player].step(time_step)
                    action_list = [agent_output.action]
                    time_step = self._envs[player].step(action_list)

                # Episode is over, step all agents with final info state.
                self._agents[player].step(time_step)


    def __call__(self, game, distributions):
        """
        Call the RL oracle to find the best response strategy for each player.
        :param game: A MFG.
        :param distributions: the MFG distribution an agent best responds to.
        :return:
        """
        for player in range(self._num_players):
            self._envs[player]._distribution = distributions[player]


