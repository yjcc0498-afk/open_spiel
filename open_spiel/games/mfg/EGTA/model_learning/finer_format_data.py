from open_spiel.python.policy import get_tabular_policy_states
import numpy as np

def sort_tuple(tup):
    tup.sort(key=lambda x: (x[0][1], x[0][0]))
    return tup

def reshape(li, n, m):
    """
    Reshape a list to n rows and m cols.
    """
    results = []
    for i in range(n):
       result = []
       for j in range(m):
           result.append(li[i*n+j])
       results.append(result)

    return results

class Formattor():
    def __init__(self, mfg_game, size, horizon, two_dim=False):
        self._mfg_game = mfg_game
        self._horizon = horizon
        self._size = size

        self._states = get_tabular_policy_states(mfg_game)
        # print("s:", self._states)
        self._state_str = []
        for t in np.arange(self._horizon):
            state_str_per_time = []
            if two_dim:
                for i in np.arange(size):
                    for j in np.arange(size):
                        state_str_per_time.append(str((i, j, t)))
            else:
                for s in np.arange(size):
                    state_str_per_time.append(str((s,t)))
            self._state_str.append(state_str_per_time)

        # print(self._state_str)

    def formatting(self, policy, distribution):
        policy_format = self.format_policy(policy)
        dist_format = self.format_distribution(distribution)

        return np.concatenate((dist_format, policy_format), axis=1)


    def format_policy(self, policy):
        format_over_time = []
        for state_strs_per_time in self._state_str:
            format = []
            for state_str in state_strs_per_time:
                state = self._states[state_str]
                format.extend(policy(state).values()) # Check if probs are ordered by action.
            format_over_time.append(format)
        return format_over_time

    def format_distribution(self, distribution):
        format_over_time = []
        for state_strs_per_time in self._state_str:
            format = []
            for state_str in state_strs_per_time:
                state = self._states[state_str]
                state_str = str(state)
                format.append(distribution.value_str(state_str))
            format_over_time.append(format)

        return format_over_time