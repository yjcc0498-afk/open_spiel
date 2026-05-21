from sklearn import datasets
from open_spiel.python.mfg.algorithms.EGTA.model_learning.hk_regression import hk_model_regression, create_mini_batches
import numpy as np
import jax

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# X, Y = datasets.load_boston(return_X_y=True)
# Y = Y[..., None]
# print(np.shape(X), np.shape(Y), type(X), type(Y))
# print(X[:5])
# print(Y[:5])


# X = np.linspace(-2., 2., num=128)[:, None]  # Generate array of shape (128, 1).
# Y = X ** 2
# print(np.shape(X), np.shape(Y), type(X), type(Y))

# X = [[1,2,3], [4,5,6], [7,8,9],[1,2,3], [4,5,6], [7,8,9],[1,2,3], [4,5,6], [7,8,9]]
# Y = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
#
# mini_batches = create_mini_batches(X, Y, 3)
# print(mini_batches[0][0])
# print(mini_batches[0][1])


# nn_params = {}
# nn_params['learning_rate'] = 0.003
# nn_params['epoch'] = 10000
# nn_params['batch_size'] = 32
# # nn_params['batch_size'] = X.shape[0]
# # nn_params['output_sizes'] = [128, 128, 1]
# nn_params['output_sizes'] = [13, 10, 15, 1]
#
# model, params = hk_model_regression(data=X, labels=Y, nn_params=nn_params, verbose=True)
#
# rng = jax.random.PRNGKey(42)
#
# plt.scatter(X, Y, label='Data')
# plt.scatter(X, model.apply(params, rng, X), label='Model prediction')
# plt.legend()
# plt.show()


def format_data(policy_idx, mixed_policy_weights, pol_value):
    one_hot_policy = np.zeros(3)
    one_hot_policy[policy_idx] = 1
    sample = np.append(one_hot_policy, mixed_policy_weights)
    sample = np.append(sample, pol_value)
    print(sample)

format_data(1, np.array([0.1, 0.3, 0.6]), 5)