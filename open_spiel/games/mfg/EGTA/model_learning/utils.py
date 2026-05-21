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


def slice_data(data, num_policies, samples_per_policy):
    """
    Pick the data for first num_policies.
    :param data:
    :param num_policies: number of policies retained.
    :param samples_per_policy: the number of samples per pure strategy.
    :return:
    """
    total_rows = num_policies * samples_per_policy
    output = data[:total_rows]
    return output


