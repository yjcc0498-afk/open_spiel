"""
A python scripts for generating all batch files for parameter tuning
"""
import shutil
import os
import itertools
import numpy as np
import itertools
import copy


ORIGIN = './base_slurm.sh'
# BASE = 'python egta_example.py --game_name=mfg_crowd_modelling --meta_strategy_method=RD --oracle_type=BR --egta_iterations=25'
# ROOT = '--root_result_folder=baseline_mfg_crowd_modelling_RD'
BASE = 'python egta_example.py --game_name=mfg_crowd_modelling_2d --meta_strategy_method=RD --oracle_type=BR --egta_iterations=25'
ROOT = '--root_result_folder=baseline_mfg_crowd_modelling_2d_RD'
SIZE = '--game_size='
HORIZON = '--game_horizon='
IL_ = '--IL_iterations='

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        raise ValueError(path + " already exists.")
    else:
        os.makedirs(path)
        print(path + " has been created successfully.")

def copy_file(original, target):
    shutil.copyfile(original, target)

def write_line(file, line):
    file.write(line)

def bash_factory():
    T = [10, 20, 30]
    size = [10, 50, 100]
    IL = [50, 100, 150, 200]

    params = list(itertools.product(T, size, IL))

    for i, param in enumerate(params):
        t, s, l = param
        target = './scripts/z_' + str(i) + '.sh'
        copy_file(ORIGIN, target)

        with open(target, 'a') as file:
            commands = [BASE, SIZE + str(s), HORIZON + str(t), IL_ + str(l),
                        ROOT + str(l) + '_size' + str(s) + '_step' + str(t)]
            new_command = " ".join(commands)
            write_line(file, '\n')
            write_line(file, new_command)


def bash_factory_2d():
    T = [10, 20, 30]
    IL = [50, 100, 150, 200]

    params = list(itertools.product(T, IL))

    for i, param in enumerate(params):
        t, l = param
        target = './scripts_2d/z_' + str(i) + '_2d.sh'
        copy_file(ORIGIN, target)

        with open(target, 'a') as file:
            commands = [BASE, SIZE + str(10), HORIZON + str(t), IL_ + str(l),
                        ROOT + str(l) + '_step' + str(t)]
            new_command = " ".join(commands)
            write_line(file, '\n')
            write_line(file, new_command)


if __name__ == '__main__':
    # bash_factory()
    bash_factory_2d()



