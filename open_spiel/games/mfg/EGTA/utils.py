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

import numpy as np
from scipy.stats import dirichlet


def uniform_sampling_simplex(dim, num_of_points=1):
    """
    Uniformly sample points from a simplex.
    :param dim: Dimension of the simplex.
    :param num_of_points: Number of samples in total.
    :return:
    """
    zeros = np.zeros((num_of_points, 1))
    ones = np.ones((num_of_points, 1))
    samples = np.random.uniform(low=0.0, high=1.0, size=(num_of_points, dim-1))
    samples = np.append(zeros, samples, axis=1)
    samples = np.append(samples, ones, axis=1)
    samples.sort(axis=1)
    samples = np.diff(samples, axis=1)
    return samples

def dirichlet_sampling_simplex(dim, num_of_points=1, alpha_coef=1.0):
    """
    Sample on a simplex using a dirichlet distribution.
    :param dim: dimension of the points.
    :param num_of_points: number of points.
    :param alpha_coef: parameter for dirichlet distribution.
    :return:
    """
    alpha = np.ones(dim) * alpha_coef
    samples = dirichlet.rvs(size=num_of_points, alpha=alpha)
    return samples


def list_to_txt(path, list):
    with open(path, 'w') as file:
        for item in list:
            file.write("%s\n" % item)


# dim = 25
# num_points = 150

# vertice = [[0.0] * dim]
# for i in range(dim):
#     new_vert = [0.0] * dim
#     new_vert[i] = 1.0
#     vertice.append(new_vert)
#
# params = np.arange(0.05, 1.05, 0.05)
# max_list = []
# print(len(params))
# print(params)
#
# for param in params:
#     samples = dirichlet_sampling_simplex(dim, num_points, alpha_coef=param)
#     max_list.append(np.max(samples))
#
# print(max_list)


# samples = dirichlet_sampling_simplex(dim, num_points, alpha_coef=0.05)
# print(np.round(samples[:6], decimals=2))

# params = np.arange(0.01, 2.01, 0.1)
# samples = []
# for param in params:
#     # samples.append(dirichlet_sampling_simplex(dim=3, num_of_points=2, alpha_coef=param))
#     samples += list(dirichlet_sampling_simplex(dim=3, num_of_points=2, alpha_coef=param))
#
# samples = np.array(samples)
# print(samples)
# for i, mixed_policy_weights in enumerate(samples):
#     print(i, mixed_policy_weights)