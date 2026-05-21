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

import haiku as hk
import jax
from jax import value_and_grad, jit
import jax.numpy as jnp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def hk_model_regression(data,
                     labels,
                     nn_params,
                     verbose=False):

        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=0.8, random_state=123)

        # Training parameters
        rng = jax.random.PRNGKey(42)
        learning_rate = jnp.array(nn_params['learning_rate'])
        epoch = nn_params['epoch']
        batch_size = nn_params['batch_size']

        # Set up model.
        def FeedForward(x):
            mlp = hk.nets.MLP(output_sizes=nn_params['output_sizes'])
            return mlp(x)

        model = hk.transform(FeedForward)
        params = model.init(rng, X_train)

        def MeanSquaredErrorLoss(weights, input, output):
            preds = model.apply(weights, rng, input)
            return jnp.power(output - preds, 2).mean()

        @jax.jit
        def update(learning_rate, params, X, Y):
            loss, param_grads = value_and_grad(MeanSquaredErrorLoss)(params, X, Y)
            return jax.tree_multimap(
                lambda p, g: p - learning_rate * g, params, param_grads
            ), loss

        batches = create_mini_batches(X_train, Y_train, batch_size)
        for i in range(1, epoch + 1):
            losses = []  # Record loss of each batch.
            for batch in batches:
                X_batch, Y_batch = batch[0], batch[1]  # Single batch of data
                params, loss = update(learning_rate, params, X_batch, Y_batch)
                losses.append(loss)  # Record Loss

            if verbose:
                if i % 100 == 0:  # Print MSE every 100 epochs
                    print("MSE : {:.2f}".format(jnp.array(losses).mean()))

        if verbose:
            train_preds = model.apply(params, rng, X_train)
            test_preds = model.apply(params, rng, X_test)
            print("Test  MSE Score : {:.2f}".format(MeanSquaredErrorLoss(params, X_test, Y_test)))
            print("Train MSE Score : {:.2f}".format(MeanSquaredErrorLoss(params, X_train, Y_train)))
            print("Test  R^2 Score : {:.2f}".format(r2_score(test_preds.squeeze(), Y_test.squeeze())))
            print("Train R^2 Score : {:.2f}".format(r2_score(train_preds.squeeze(), Y_train.squeeze())))


        return model, params


# Function to create a list containing mini-batches
def create_mini_batches(X, Y, batch_size):
    mini_batches = []
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


















