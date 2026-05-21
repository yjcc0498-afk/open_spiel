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

from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv

import functools
print = functools.partial(print, flush=True)

def fp_benchmark(mfg_game, num_iterations, writer):
    fp = fictitious_play.FictitiousPlay(mfg_game)
    print("================================")
    print("Beginning running FP benchmark.")
    for j in range(num_iterations):
        fp.iteration()
        fp_policy = fp.get_policy()
        nash_conv_fp = nash_conv.NashConv(mfg_game, fp_policy)
        nashconv_value = nash_conv_fp.nash_conv()
        writer.add_scalar('fp_exp', nashconv_value, j)
        print("Iteration : {}".format(j), "NashConv : {}".format(nashconv_value))


def md_benchmark(mfg_game, num_iterations, writer):
    md = mirror_descent.MirrorDescent(mfg_game)
    print("================================")
    print("Beginning running MD benchmark.")
    for j in range(num_iterations):
        md.iteration()
        md_policy = md.get_policy()
        nash_conv_md = nash_conv.NashConv(mfg_game, md_policy)
        nashconv_value = nash_conv_md.nash_conv()
        writer.add_scalar('md_exp', nashconv_value, j)
        print("Iteration : {}".format(j), "NashConv : {}".format(nashconv_value))