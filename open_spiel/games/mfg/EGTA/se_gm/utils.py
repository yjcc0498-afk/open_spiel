import numpy as np
from cvxopt import matrix, solvers


"""
Project n-dim vector onto n-1 dim.
"""

def qp_projection(mixed_strategy):
    """
    Follow https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    """
    n = len(mixed_strategy) - 1
    P = matrix(np.eye(n), tc='d')
    q = matrix(-np.array(mixed_strategy[:-1]), tc='d')
    G = matrix(-np.eye(n), tc='d')
    h = matrix(np.zeros(n), tc='d')
    A = matrix(np.ones(n)[None, ...], tc='d')
    b = matrix(np.array([1]), tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    x = np.array(sol['x'])
    x_flat = np.squeeze(x)
    x_flat = np.pad(x_flat, (0, 1), 'constant')
    return x_flat


def qp_projection2(mixed_strategy):
    """ Same results as qp_projection. """
    n = len(mixed_strategy)
    if mixed_strategy[-1] == 0 or n == 1:
        return mixed_strategy
    new_strategy = np.copy(mixed_strategy)
    new_strategy += new_strategy[-1] / (n-1)
    new_strategy[-1] = 0
    return new_strategy


def simplex_projection(mixed_strategy):
    n = len(mixed_strategy)
    if mixed_strategy[-1] == 0 or n == 1:
        return mixed_strategy
    deno = np.sum(mixed_strategy[:-1])
    mixed_strategy /= deno
    mixed_strategy[-1] = 0
    return mixed_strategy



