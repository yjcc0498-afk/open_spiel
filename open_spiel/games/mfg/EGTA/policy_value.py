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

"""Does a backward pass to output the value of a policy."""
import collections
import numpy as np

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
import pyspiel


class PolicyValue(value.ValueFunction):
  """Computes the value of a specified strategy."""

  def __init__(self,
               game,
               distribution: distribution_std.Distribution,
               policy: policy_std.Policy,
               root_state=None,
               evaluation=True,
               sims_per_entry=None):
    """Initializes the value calculation.

    Args:
      game: The game to analyze.
      distribution: A `distribution.Distribution` object.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start. If `None`, the game
        root state is used.
    """
    super(PolicyValue, self).__init__(game)
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._distribution = distribution
    self._policy = policy
    self._sims_per_entry = sims_per_entry
    self._use_sampling = False

    self._state_value = collections.defaultdict(float)

    if sims_per_entry:
        self._use_sampling = True
        self._sampled_states = collections.defaultdict(bool)

    if evaluation:
      self.evaluate()

  def eval_state(self, state):
    """Evaluate the value of a state."""
    state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    if state_str in self._state_value:
      return self._state_value[state_str]
    elif state.is_terminal():
      self._state_value[state_str] = state.rewards()[
          state.mean_field_population()]
      return self._state_value[state_str]
    elif state.current_player() == pyspiel.PlayerId.CHANCE:
      self._state_value[state_str] = 0.0
      for action, prob in state.chance_outcomes():
        new_state = state.child(action)
        self._state_value[state_str] += prob * self.eval_state(new_state)
      return self._state_value[state_str]
    elif state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      dist_to_register = state.distribution_support()
      dist = [
          self._distribution.value_str(str_state, 0.)
          for str_state in dist_to_register
      ]
      new_state = state.clone()
      new_state.update_distribution(dist)
      self._state_value[state_str] = (
          state.rewards()[state.mean_field_population()] +
          self.eval_state(new_state))
      return self._state_value[state_str]
    else:
      assert int(state.current_player()) >= 0, "The player id should be >= 0"
      v = 0.0
      for action, prob in self._policy.action_probabilities(state).items():
        new_state = state.child(action)
        v += prob * self.eval_state(new_state)
      self._state_value[state_str] = state.rewards()[
          state.mean_field_population()] + v
      return self._state_value[state_str]

  def evaluate(self):
    """Evaluate the value over states of self._policy."""
    for state in self._root_states:
      if self._sims_per_entry is None:
        self.eval_state(state)
      else:
        self.eval_state_with_sampling(state)

  def value(self, state, action=None):
    if action is None:
      if self._use_sampling and not self._sampled_states[state]:
          raise ValueError("The state has not been evaluated by sampling.")
      return self._state_value[state.observation_string(
          pyspiel.PlayerId.DEFAULT_PLAYER_ID)]
    else:
      new_state = state.child(action)
      return state.rewards()[0] + self._state_value[
          new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)]


  # Estimate policy value by sampling.
  # TODO: Does this work for multiple population?
  def eval_state_with_sampling(self, state):
      """
      Estimate the value of a state with sampling.
      The purpose of having this function is to reduce the evaluation cost on a profile.
      In large games, fully evaluating a policy is expensive.
      """
      assert isinstance(self._sims_per_entry, int)

      state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
      total_reward = 0
      for _ in range(self._sims_per_entry):
        new_state = state.clone()
        total_reward += self.sample_episode(new_state)

      self._sampled_states[state_str] = True
      self._state_value[state_str] = total_reward / self._sims_per_entry


  def sample_episode(self, state):
      """Sample an episode."""
      cumulative_reward = 0
      while not state.is_terminal():
          if state.current_player() == pyspiel.PlayerId.CHANCE:
              action_list, prob_list = zip(*state.chance_outcomes())
              action = np.random.choice(action_list, p=prob_list)
              state.apply_action(action)
          elif state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
              dist_to_register = state.distribution_support()
              dist = [
                  self._distribution.value_str(str_state, 0.)
                  for str_state in dist_to_register
              ]
              state.update_distribution(dist)
          else:
              assert int(state.current_player()) >= 0, "The player id should be >= 0"
              cumulative_reward += state.rewards()[state.mean_field_population()]
              action_list, prob_list = zip(*self._policy.action_probabilities(state).items())
              action = np.random.choice(action_list, p=prob_list)
              state.apply_action(action)

      if state.is_terminal():
          cumulative_reward += state.rewards()[state.mean_field_population()]

      return cumulative_reward