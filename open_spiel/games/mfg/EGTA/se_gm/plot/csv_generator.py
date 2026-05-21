import numpy as np
from numpy import genfromtxt, savetxt


# base_name = './data_1d/RD30'
# base_name = './data_2d/RD30'


sizes = [100, 50, 10]
steps = [10, 20, 30]

# for s in sizes:
#     for t in steps:
#         for i in range(1, 6):
#             name = base_name + '_size' + str(s) + '_step' + str(t) + '_' + str(i) + '.csv'
#             with open(name, 'w') as file:
#                 pass

base_name = './baselines_1d/cd_egta'

for s in sizes:
    for t in steps:
        name = base_name + '_size' + str(s) + '_step' + str(t) + '_consistent' + '.csv'
        with open(name, 'w') as file:
            pass



# for t in steps:
#     for i in range(1, 6):
#         name = base_name + '_step' + str(t) + '_' + str(i) + '.csv'
#         with open(name, 'w') as file:
#             pass


# for s in sizes:
#     for t in steps:
#         data = []
#         for i in range(1, 6):
#             name = base_name + '_size' + str(s) + '_step' + str(t) + '_' + str(i) + '.csv'
#             RD = genfromtxt(name, delimiter=',')[:25]
#             data.append(RD)
#
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#         savetxt('./data_1d/RD30' + '_size' + str(s) + '_step' + str(t) + '_mean' + '.csv', mean, delimiter=",")
#         savetxt('./data_1d/RD30' + '_size' + str(s) + '_step' + str(t) + '_std' + '.csv', std, delimiter=",")



# for t in steps:
#     data = []
#     for i in range(1, 6):
#         name = base_name + '_step' + str(t) + '_' + str(i) + '.csv'
#         RD = genfromtxt(name, delimiter=',')[:25]
#         data.append(RD)
#
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     savetxt('./data_2d/RD30' + '_step' + str(t) + '_mean' + '.csv', mean, delimiter=",")
#     savetxt('./data_2d/RD30' + '_step' + str(t) + '_std' + '.csv', std, delimiter=",")