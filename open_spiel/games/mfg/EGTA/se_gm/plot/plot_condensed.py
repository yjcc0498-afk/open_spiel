from numpy import genfromtxt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


egta = [i * 200 for i in range(26)]
egta = np.cumsum(egta)
egta_model = [i * 30 for i in range(26)]
egta_model = np.cumsum(egta_model)

X = np.arange(0, 26).astype(dtype=np.str)

plt.xticks(np.arange(0, 26, 5), size = 19)
plt.yticks(size = 19)

plt.plot(X, egta, color="C0", label='EGTA w. simulator only')
plt.plot(X, egta_model, color="C1", label='EGTA w. GML/REG')


plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Number of Simulations', fontsize = 21)

plt.legend(loc="best", prop={'size': 20})

# plt.title("Size=" + str(size) + ", Horizon=" + str(step), fontsize=20)

plt.tight_layout()
plt.savefig('./figures/num_simulations.png')
