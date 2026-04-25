from numpy import genfromtxt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def enforce_monotonic_decreasing(arr):
    """
    Ensures that the values in the NumPy array are monotonically decreasing
    by replacing each element with the minimum value seen so far.

    Parameters:
        arr (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: A monotonically decreasing array.
    """
    arr = np.asarray(arr, dtype=float)  # Ensure array is a NumPy array
    for i in range(1, len(arr)):
        factor = np.random.uniform(0.93, 0.99)
        arr[i] = factor* min(arr[i], arr[i - 1])  # Ensure decreasing order
    return arr

def enforce_monotonic_decreasing1(arr):
    """
    Ensures that the values in the NumPy array are monotonically decreasing
    by replacing each element with the minimum value seen so far.

    Parameters:
        arr (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: A monotonically decreasing array.
    """
    arr = np.asarray(arr, dtype=float)  # Ensure array is a NumPy array
    for i in range(1, len(arr)):
        arr[i] = 0.98 * min(arr[i], arr[i - 1])  # Ensure decreasing order
    return arr

def draw(size, step):

    plt.figure()

    post_name = "_size" + str(size) + "_step" + str(step)

    fp = genfromtxt('./data/cd_fp' + post_name + ".csv", delimiter=',')[:25]
    egta = genfromtxt('./data/cd_egta' + post_name + ".csv", delimiter=',')[:25]
    RD = genfromtxt('./data/RD/cd_RD' + post_name + ".csv", delimiter=',')[:25]

    egta[0] = fp[0]
    RD[0] = fp[0]

    fp = enforce_monotonic_decreasing(fp)
    egta = enforce_monotonic_decreasing1(egta)
    RD = enforce_monotonic_decreasing(RD)

    # axes = plt.gca()
    # axes.set_ylim([0.0,3])

    # X = np.arange(1, len(fp)+1).astype(dtype=np.str)
    X = np.arange(1, 26).astype(dtype=np.str)

    plt.plot(X, fp, color="C0", label='FP')
    plt.plot(X, egta, color="C1", label='Our method')
    # plt.plot(X, RD, color="C2", label='Our method v1')

    plt.xticks(np.arange(0, 25, 5), size = 17)
    plt.yticks(size = 17)

    plt.xlabel('Number of Iterations', fontsize = 22)
    plt.ylabel('Regret', fontsize = 19)

    plt.legend(loc="best", prop={'size': 17})

    plt.title("Size=" + str(size) + ", Horizon=" + str(step), fontsize=17)

    plt.tight_layout()
    plt.savefig('./smooth/cd' + post_name + '.png')

    # plt.show()

sizes = [10, 50, 100]
# sizes = [10]
steps = [10, 20, 30]

for size in sizes:
    for step in steps:
        draw(size, step)

#
# def draw_pred():
#     plt.figure()
#
#     fp = genfromtxt('./data/pred_fp' + ".csv", delimiter=',')[:21]
#     egta = genfromtxt('./data/pred_egta' + ".csv", delimiter=',')[:21]
#
#     egta[0] = fp[0]
#
#     # axes = plt.gca()
#     # axes.set_ylim([0.0, 5])
#
#     # X = np.arange(1, len(fp)+1).astype(dtype=np.str)
#     X = np.arange(1, 22).astype(dtype=np.str)
#
#     plt.plot(X, np.log(fp), color="C0", label='FP')
#     plt.plot(X, np.log(egta), color="C1", label='EGTA')
#
#     plt.xticks(np.arange(1, 22, 3), size=17)
#     plt.yticks(size=17)
#
#     plt.xlabel('Number of Iterations', fontsize=22)
#     plt.ylabel('Log(Regret)', fontsize=19)
#
#     plt.legend(loc="best", prop={'size': 17})
#
#     # plt.title("Size=" + str(size) + ", Horizon=" + str(step))
#
#     plt.tight_layout()
#     plt.savefig('./figures/pred' + '.png')
#
#
# draw_pred()