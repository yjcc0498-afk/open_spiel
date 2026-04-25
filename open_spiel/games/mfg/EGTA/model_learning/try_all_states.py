from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.EGTA.model_learning.finer_format_data import Formattor
import pyspiel


def sort_tuple(tup):
    tup.sort(key=lambda x: (x[1], x[0]))
    return tup

def reshape(li, n, m):
    """
    Reshape a list to n rows and m cols.
    """
    results = []
    for i in range(m):
       result = []
       for j in range(n):
           result.append(li[i*n+j])
       results.append(result)

    return results


game = pyspiel.load_game("mfg_crowd_modelling_2d", {"size": 3, "horizon": 2})
# game = crowd_modelling.MFGCrowdModellingGame({"size": 5, "horizon": 2})
# print(game.max_game_length())

# states = get_tabular_policy_states(game)
# for k in states:
#     print(k, states[k].information_state_string())

formattor = Formattor(game, size=3, horizon=2, two_dim=True)
policy = policy_std.UniformRandomPolicy(game).to_tabular()
distrib = distribution.DistributionPolicy(game, policy)

# print(policy.state_lookup.keys())
# print(policy.game_type.provides_information_state_string)

# policy_format = formattor.format_policy(policy)
dist_format = formattor.format_distribution(distrib)
# overall_format = formattor.formatting(policy, distrib)
#
# print("policy_format:", policy_format)
print("dist_format:", dist_format)
# print("overall_format:", overall_format)


