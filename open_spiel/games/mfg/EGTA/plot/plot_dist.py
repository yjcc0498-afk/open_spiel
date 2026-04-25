from numpy import genfromtxt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import wasserstein_distance
import os

def draw(model=False, transformer=False):
    # 获取脚本文件所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    true_dist_path = os.path.join(script_dir, "Distributions", "mfg_crowd_modelling_1d_FP_dist_eval", "distribution_true.npy")
    model_dist_path = os.path.join(script_dir, "Distributions", "mfg_crowd_modelling_1d_FP_dist_eval", "distribution_model.npy")
    transformer_dist_path = os.path.join(script_dir, "Distributions", "mfg_crowd_modelling_1d_FP_dist_eval", "distribution_transformer_model.npy")
    save_path = os.path.join(script_dir, "Distributions", "dist_figures_new_all")
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    true_distribution = np.load(true_dist_path)
    model_distribution = np.load(model_dist_path)
    if transformer:
        try:
            transformer_distribution = np.load(transformer_dist_path)
        except FileNotFoundError:
            print(f"Transformer 模型分布文件不存在: {transformer_dist_path}")
            return

    # Creating dataset
    # for i, idx in enumerate([5, 10, 15, 20, 25, 29]):
    for idx in np.arange(30):
    # for idx in range(1):
    #     idx = 25
        if transformer:
            Z = transformer_distribution[idx]
        elif model:
            Z = model_distribution[idx]
        else:
            Z = true_distribution[idx]


        X = np.arange(1, 11)
        Y = np.arange(1, 11)
        X, Y = np.meshgrid(X, Y)

        Z = np.reshape(Z, (10, 10))
        # print(np.shape(Z))/

        # Plot the surface.
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)


        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title("Time Step:" + str(idx+1), fontsize=37)
        plt.xlabel('X', fontsize=28, labelpad=13)
        plt.ylabel('Y', fontsize=28, labelpad=15)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)

        plt.tight_layout()
        if transformer:
            plt.savefig(save_path + "cd_1d_transformer_t_" + str(idx) + '.png')
        elif model:
            plt.savefig(save_path + "cd_1d_model_t_" + str(idx) + '.png')
        else:
            plt.savefig(save_path + "cd_1d_true_t_" + str(idx) + '.png')
        # plt.show()


draw(model=False)
draw(model=True)
draw(transformer=True)


# def compute_Wdist():
#     true_dist_path = "./Distributions/mfg_crowd_modelling_2d_FP_dist_eval/distribution_true.npy"
#     model_dist_path = "./Distributions/mfg_crowd_modelling_2d_FP_dist_eval/distribution_model.npy"
#
#     true_distribution = np.load(true_dist_path)
#     model_distribution = np.load(model_dist_path)
#     W_dist = []
#     # for idx in [5, 10, 15, 20, 25, 29]:
#     for idx in np.arange(30):
#         true_dist = true_distribution[idx]
#         model_dist = model_distribution[idx]
#
#         W_dist.append(np.around(wasserstein_distance(true_dist, model_dist), decimals=5))
#
#     print(np.reshape(W_dist, (3,10)))
#
# compute_Wdist()

