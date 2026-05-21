from numpy import genfromtxt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os


def draw(size, step):

    plt.figure()

    post_name = "_size" + str(size) + "_step" + str(step)

    # 1d
    # base_name = './data_1d/RD30'
    # mean_name = base_name + '_size' + str(size) + '_step' + str(step) + '_mean.csv'
    # std_name = base_name + '_size' + str(size) + '_step' + str(step) + '_std.csv'

    # 2d
    base_name = './data_2d/RD30'
    mean_name = base_name + '_step' + str(step) + '_mean.csv'
    std_name = base_name + '_step' + str(step) + '_std.csv'
    print(os.getcwd())

    # fp = genfromtxt('./baselines_1d/cd_fp' + post_name + ".csv", delimiter=',')[:25]
    # FP = genfromtxt('../../plot/data/cd_egta' + post_name + ".csv", delimiter=',')[:25]
    # RD = genfromtxt('./baselines_1d/cd_egta' + post_name + ".csv", delimiter=',')[:25]

    fp = genfromtxt('./baselines_2d/cd_2d_fp' + post_name + ".csv", delimiter=',')[:25]
    FP = genfromtxt('../../plot/data/cd_2d_egta' + post_name + ".csv", delimiter=',')[:25]
    egta = genfromtxt('./baselines_2d/cd_2d_egta' + post_name + ".csv", delimiter=',')[:25]

    RD_model_mean = genfromtxt(mean_name, delimiter=',')
    RD_model_std = genfromtxt(std_name, delimiter=',')
    # print(RD_std)

    # RD_mean = np.log(RD_mean)
    # RD_std = np.log(RD_std)

    RD[0] = fp[0]
    FP[0] = fp[0]
    RD_model_mean[0] = fp[0]

    # axes = plt.gca()
    # axes.set_ylim([0.0,3])

    # X = np.arange(1, len(fp)+1).astype(dtype=np.str)
    X = np.arange(1, 26).astype(dtype=np.str)

    plt.plot(X, fp, color="C0", label='FP')
    plt.plot(X, FP, color="C1", label='EGTA w. FP')
    plt.plot(X, RD, color="C2", label='EGTA w. RD')


    plt.plot(X, RD_model_mean, color="C3", label='EGTA w. GML')
    plt.fill_between(X, RD_model_mean + RD_model_std, RD_model_mean - RD_model_std, alpha=0.3, color="C3")

    plt.xticks(np.arange(0, 26, 5), size = 19)
    plt.yticks(size = 19)

    plt.xlabel('Number of Iterations', fontsize = 22)
    plt.ylabel('Regret', fontsize = 21)

    plt.legend(loc="best", prop={'size': 20})

    plt.title("Size=" + str(size) + ", Horizon=" + str(step), fontsize=20)

    plt.tight_layout()
    # plt.savefig('./combined_all/cd' + post_name + '.png')
    plt.savefig('./combined_all/cd_2d' + post_name + '.png')

    # plt.show()

# sizes = [10, 50, 100]
sizes = [10]
steps = [10, 20, 30]

for size in sizes:
    for step in steps:
        draw(size, step)