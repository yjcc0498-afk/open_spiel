from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play, mirror_descent
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling, predator_prey
import pyspiel

def run_fp_crowdmodelling():
    game = crowd_modelling.MFGCrowdModellingGame()
    fp = fictitious_play.FictitiousPlay(game)
    print("Start Iterations")
    for i in range(50):
        print("Iter:", i+1)
        fp.iteration()
        fp_policy = fp.get_policy()
        nash_conv_fp = nash_conv.NashConv(game, fp_policy)
        print("nash_conv_fp:", nash_conv_fp.nash_conv())
    print("End Iterations")
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(game, fp_policy)
    print("nash_conv_fp:", nash_conv_fp.nash_conv())
    
    
def run_md_crowdmodelling():
    game = crowd_modelling.MFGCrowdModellingGame()
    md = mirror_descent.MirrorDescent(game)
    print("Start Iterations")
    for i in range(50):
        print("Iter:", i+1)
        md.iteration()
        md_policy = md.get_policy()
        nash_conv_md = nash_conv.NashConv(game, md_policy)
        print("nash_conv_md:", nash_conv_md.nash_conv())
    print("End Iterations")
    md_policy = md.get_policy()
    nash_conv_md = nash_conv.NashConv(game, md_policy)
    print("nash_conv_md:", nash_conv_md.nash_conv())


def run_fp_pred():
    game = predator_prey.MFGPredatorPreyGame()
    fp = fictitious_play.FictitiousPlay(game)
    num_players = game.num_players()
    print("num_players:", num_players)
    print("Start Iterations")
    for i in range(5):
        print("Iter:", i+1)
        fp.iteration()
    print("End Iterations")
    fp_policy = fp.get_policy()
    nash_conv_fp = nash_conv.NashConv(game, fp_policy)
    print("nash_conv_fp:", nash_conv_fp.nash_conv())

def check_pred():
    game = predator_prey.MFGPredatorPreyGame()
    # game = crowd_modelling.MFGCrowdModellingGame()
    print("num_players:", game.num_players())
    root_state = []
    for player in range(game.num_players()):
        print("Current player:", player)
        root_state.append(game.new_initial_state_for_population(population=player))

    _root_states = game.new_initial_states()
    print("_root_states", _root_states)
    print(len(root_state), len(_root_states))

    for state in _root_states:
        print(state._population, type(state._population))

def check_load_game():
    game1 = pyspiel.load_game("mfg_crowd_modelling", {"size": 50, "horizon": 20})
    game2 = pyspiel.load_game("mfg_crowd_modelling_2d")
    game3 = pyspiel.load_game("python_mfg_crowd_modelling")
    game4 = pyspiel.load_game("python_mfg_predator_prey")
    game5 = pyspiel.load_game("python_mfg_pdtor_prey")
    print(game1, game3, game4)

# run_fp_crowdmodelling()
# run_md_crowdmodelling()
# run_fp_pred()
# check_pred()
check_load_game()