import numpy as np
import itertools
from open_spiel.python.mfg.algorithms.EGTA.utils import list_to_txt

# for i in range(12):
#     print('sbatch z_' + str(i) + '_2d.sh')


# def list_to_txt(path, list):
#     with open(path, 'w') as file:
#         for item in list:
#             file.write("%s\n" % item)


path = "./hello.txt"
list = [0.1, 0.2, 0.3]

list_to_txt(path, list)