import collections
import math
import re

import numpy as np
import pyspiel


STATE_PATTERN_1D = re.compile(r"^\(([-\d]+),\s*([-\d]+)\)")
STATE_PATTERN_2D = re.compile(r"^\(([-\d]+),\s*([-\d]+),\s*([-\d]+)\)")


def _type_from_states(states):
    types = [state.get_type() for state in states]
    assert len(set(types)) == 1, "Expected homogeneous state types, got {}".format(types)
    return types[0]


def _state_to_str(state):
    return state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)


def _forward_actions(current_states, distribution, actions_and_probs_fn):
    new_states = []
    new_distribution = collections.defaultdict(float)
    for state in current_states:
        state_str = _state_to_str(state)
        state_weight = distribution.get(state_str, 0.0)
        for action, prob in actions_and_probs_fn(state):
            new_state = state.child(action)
            new_state_str = _state_to_str(new_state)
            if new_state_str not in new_distribution:
                new_states.append(new_state)
            new_distribution[new_state_str] += prob * state_weight
    return new_states, new_distribution


def _one_forward_step(current_states, distribution, policy):
    state_type = _type_from_states(current_states)
    if state_type == pyspiel.StateType.CHANCE:
        return _forward_actions(current_states, distribution, lambda state: state.chance_outcomes())

    if state_type == pyspiel.StateType.MEAN_FIELD:
        new_states = []
        new_distribution = {}
        for state in current_states:
            dist = [distribution.get(str_state, 0.0) for str_state in state.distribution_support()]
            new_state = state.clone()
            new_state.update_distribution(dist)
            new_state_str = _state_to_str(new_state)
            if new_state_str not in new_distribution:
                new_states.append(new_state)
                new_distribution[new_state_str] = 0.0
            new_distribution[new_state_str] += distribution.get(_state_to_str(state), 0.0)
        return new_states, new_distribution

    if state_type == pyspiel.StateType.DECISION:
        return _forward_actions(
            current_states,
            distribution,
            lambda state: policy.action_probabilities(state).items(),
        )

    raise ValueError("Unexpected state type {}".format(state_type))


class StrategyFeatureExtractor(object):
    """Extracts [T, F] behavior fingerprints for a pure policy."""

    def __init__(self, game, sequence_length=None, epsilon=1e-12):
        self._game = game
        self._sequence_length = sequence_length or game.max_game_length()
        self._epsilon = epsilon

        self._root_states = game.new_initial_states()
        self._state_dim = None
        self._goal_point = None
        self.feature_names = [
            "expected_action",
            "action_var",
            "action_entropy",
            "negative_action_mass",
            "zero_action_mass",
            "positive_action_mass",
            "state_mean",
            "state_var",
            "state_std",
            "state_concentration",
            "time_ratio",
        ]

    @property
    def feature_dim(self):
        return len(self.feature_names)

    def extract_features(self, policy):
        current_states = list(self._root_states)
        current_distribution = {_state_to_str(state): 1.0 for state in current_states}
        features = []

        current_states, current_distribution = self._advance_to_decision_states(
            current_states, current_distribution, policy
        )

        for t in range(self._sequence_length):
            if not current_states or _type_from_states(current_states) == pyspiel.StateType.TERMINAL:
                break
            features.append(self._compute_step_features(current_states, current_distribution, policy, t))
            current_states, current_distribution = self._advance_to_next_decision_step(
                current_states, current_distribution, policy
            )

        while len(features) < self._sequence_length:
            pad_t = len(features)
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[-1] = self._time_ratio(pad_t)
            features.append(padded)

        return np.asarray(features, dtype=np.float32)

    def batch_extract(self, policies):
        return np.asarray([self.extract_features(policy) for policy in policies], dtype=np.float32)

    def _advance_to_decision_states(self, current_states, current_distribution, policy):
        while current_states and _type_from_states(current_states) not in (
                pyspiel.StateType.DECISION, pyspiel.StateType.TERMINAL):
            current_states, current_distribution = _one_forward_step(
                current_states, current_distribution, policy
            )
        return current_states, current_distribution

    def _advance_to_next_decision_step(self, current_states, current_distribution, policy):
        if not current_states:
            return current_states, current_distribution

        current_states, current_distribution = _one_forward_step(
            current_states, current_distribution, policy
        )
        return self._advance_to_decision_states(current_states, current_distribution, policy)

    def _compute_step_features(self, current_states, current_distribution, policy, t):
        expected_action = 0.0
        expected_action_square = 0.0
        action_entropy = 0.0
        negative_action_mass = 0.0
        zero_action_mass = 0.0
        positive_action_mass = 0.0
        state_mean = 0.0
        state_concentration = 0.0
        raw_state_values = []

        for state in current_states:
            state_prob = current_distribution.get(_state_to_str(state), 0.0)
            state_value = self._state_value(state)
            raw_state_values.append((state_prob, state_value))
            state_mean += state_prob * state_value
            state_concentration += state_prob ** 2

            action_probs = policy.action_probabilities(state)
            for action, prob in action_probs.items():
                mass = state_prob * prob
                action_value = self._action_value(state, action)
                expected_action += mass * action_value
                expected_action_square += mass * (action_value ** 2)
                if action_value < 0:
                    negative_action_mass += mass
                elif action_value > 0:
                    positive_action_mass += mass
                else:
                    zero_action_mass += mass
                if prob > 0:
                    action_entropy -= mass * math.log(prob + self._epsilon)

        state_var = 0.0
        for state_prob, state_value in raw_state_values:
            state_var += state_prob * ((state_value - state_mean) ** 2)

        action_var = max(expected_action_square - (expected_action ** 2), 0.0)
        state_std = math.sqrt(max(state_var, 0.0))

        return np.array([
            expected_action,
            action_var,
            action_entropy,
            negative_action_mass,
            zero_action_mass,
            positive_action_mass,
            state_mean,
            state_var,
            state_std,
            state_concentration,
            self._time_ratio(t),
        ], dtype=np.float32)

    def _time_ratio(self, t):
        denom = max(self._sequence_length, 1)
        return float(t) / float(denom)

    def _state_value(self, state):
        coords = self._parse_state_coordinates(state)
        if len(coords) == 1:
            return float(coords[0])
        goal_x, goal_y = self._goal_coordinates(state)
        dx = coords[0] - goal_x
        dy = coords[1] - goal_y
        return float(math.sqrt(dx * dx + dy * dy))

    def _action_value(self, state, action):
        coords = self._parse_state_coordinates(state)
        if len(coords) == 1:
            return self._action_value_1d(action)

        next_state = state.child(action)
        next_coords = self._parse_state_coordinates(next_state)
        cur_dist = self._distance_to_goal(coords, state)
        next_dist = self._distance_to_goal(next_coords, next_state)
        return float(cur_dist - next_dist)

    def _action_value_1d(self, action):
        action = int(action)
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping.get(action, 0.0)

    def _distance_to_goal(self, coords, state):
        goal_x, goal_y = self._goal_coordinates(state)
        dx = abs(coords[0] - goal_x)
        dy = abs(coords[1] - goal_y)
        return dx + dy

    def _goal_coordinates(self, state):
        if self._goal_point is None:
            size = self._infer_grid_size(state)
            center = size / 2.0
            self._goal_point = (center, center)
        return self._goal_point

    def _infer_grid_size(self, state):
        support_size = len(state.distribution_support())
        if support_size <= 1:
            return 1
        return int(round(math.sqrt(float(support_size))))

    def _parse_state_coordinates(self, state):
        state_str = _state_to_str(state)
        match_2d = STATE_PATTERN_2D.match(state_str)
        if match_2d:
            self._state_dim = 2
            return (float(match_2d.group(1)), float(match_2d.group(2)))

        match_1d = STATE_PATTERN_1D.match(state_str)
        if match_1d:
            self._state_dim = 1
            return (float(match_1d.group(1)),)

        raise ValueError("Unsupported state string format: {}".format(state_str))
