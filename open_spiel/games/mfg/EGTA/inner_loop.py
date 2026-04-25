# Inner loop of EGTA implements the profile selection in an empirical game
# guided by meta-strategy solvers. Here a MFG equilibrium of the empirical game
# is being searched, which serves as a best-response target in EGTA.



class InnerLoop(object):
    """
    Computes a best-response target for EGTA.
    """

    def __init__(self, meta_strategy_method):
        """
        Run inner loop to analysis an empirical game.
        :param meta_strategy_method: a object of meta-strategy solver class.
        """

        self._meta_strategy_method = meta_strategy_method


    def run_inner_loop(self):
        """
        Run inner loop algorithm.
        :return a merged policy. (i.e., a merged empirical NE policy.)
        """
        self._meta_strategy_method.run()
        output_merged_policy = self._meta_strategy_method.get_output_policies()

        return output_merged_policy



























