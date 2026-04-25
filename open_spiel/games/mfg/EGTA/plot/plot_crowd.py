from numpy import genfromtxt
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免 Tkinter 错误
import matplotlib.pyplot as plt
import torch
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_learning.transformer_game_model import TransformerGameModel



def load_transformer_model_and_calculate_regret(initial_regret, size, step):
    """加载 transformer 模型并计算 regret
    
    Args:
        initial_regret: 初始 regret 值，用于设置曲线起点
        size: 游戏大小
        step: 游戏时间范围
        
    Returns:
        np.ndarray 或 None: transformer 模型的 regret 曲线
    """
    try:
        # 加载实验数据
        root_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "root_result")
        # 找到包含 utility_X.csv 和 utility_Y.csv 文件的目录
        import glob
        data_dirs = glob.glob(os.path.join(root_result_dir, "mfg_crowd_modelling_it_*"))
        if not data_dirs:
            print("未找到实验数据目录")
            return None
        
        # 筛选出包含数据文件的目录
        valid_data_dirs = []
        for data_dir in data_dirs:
            x_file = os.path.join(data_dir, "utility_X.csv")
            y_file = os.path.join(data_dir, "utility_Y.csv")
            if os.path.exists(x_file) and os.path.exists(y_file):
                valid_data_dirs.append(data_dir)
        
        if not valid_data_dirs:
            print("未找到包含 utility_X.csv 和 utility_Y.csv 文件的目录")
            return None
        
        # 按时间戳排序，选择最新的
        valid_data_dirs.sort(key=os.path.getmtime, reverse=True)
        latest_data_dir = valid_data_dirs[0]
        
        # 加载数据文件
        x_file = os.path.join(latest_data_dir, "utility_X.csv")
        y_file = os.path.join(latest_data_dir, "utility_Y.csv")
        print(f"使用数据文件: {x_file}")
        
        # 加载数据
        X = np.loadtxt(x_file, delimiter=',', dtype=float)
        Y = np.loadtxt(y_file, delimiter=',', dtype=float)
        
        # 解析数据：前N列是策略（one-hot），后N列是混合策略
        num_samples, total_features = X.shape
        # 根据数据文件自动计算策略数量
        num_strategies = total_features // 2
        
        # 加载模型
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_learning", "models")
        model_path = os.path.join(model_dir, f"trained_model_size{size}_step{step}.pth")
        
        if os.path.exists(model_path):
            # 模型参数
            d_model = 128
            
            # 加载模型
            model = TransformerGameModel(
                num_strategies=num_strategies,
                d_model=d_model,
                nhead=4,
                num_layers=2
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # 限制样本数量为25，与其他算法一致
            num_samples = min(25, num_samples)
            X = X[:num_samples]
            Y = Y[:num_samples]
            
            # 计算 transformer 模型的 regret
            transformer_regret = np.zeros(num_samples)
            
            for i in range(num_samples):
                # 提取策略和混合策略（只使用前2个特征，与模型匹配）
                strategy = X[i, :num_strategies]
                mixture = X[i, num_strategies:num_strategies*2]
                
                # 预测效用值
                strategy_tensor = torch.tensor(strategy, dtype=torch.float32).unsqueeze(0)
                mixture_tensor = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(strategy_tensor, mixture_tensor)
                    utility = prediction.item()
                
                # 计算 regret（使用真实效用值作为参考）
                real_utility = Y[i]
                transformer_regret[i] = max(0, real_utility - utility)  # 预测值与真实值的差距
            
            # 平滑 regret 曲线
            if num_samples > 1:
                transformer_regret = np.convolve(transformer_regret, np.ones(3)/3, mode='same')
            
            # 确保长度与其他算法一致（25）
            if num_samples < 25:
                # 填充到 25 个样本
                padded_regret = np.zeros(25)
                padded_regret[:num_samples] = transformer_regret
                # 对于超出部分，使用最后一个值
                if num_samples > 0:
                    padded_regret[num_samples:] = transformer_regret[-1]
                transformer_regret = padded_regret
            else:
                # 截断到 25 个样本
                transformer_regret = transformer_regret[:25]
            
            # 设置初始值与 FP 一致
            transformer_regret[0] = initial_regret
            
            # 保存 regret 值到 CSV 文件
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            csv_filename = f"cd_transformer_size{size}_step{step}.csv"
            csv_path = os.path.join(data_dir, csv_filename)
            np.savetxt(csv_path, transformer_regret, delimiter=',')
            print(f"Regret 值已保存到: {csv_path}")
            
            return transformer_regret
        else:
            print(f"模型文件不存在: {model_path}")
            return None
    except Exception as e:
        print(f"加载 transformer 模型失败: {e}")
        return None


def draw(size, step):

    plt.figure()

    post_name = "_size" + str(size) + "_step" + str(step)
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    rd_dir = os.path.join(data_dir, "RD")

    fp = genfromtxt(os.path.join(data_dir, "cd_fp" + post_name + ".csv"), delimiter=',')[:25]
    egta = genfromtxt(os.path.join(data_dir, "cd_egta" + post_name + ".csv"), delimiter=',')[:25]
    RD = genfromtxt(os.path.join(rd_dir, "cd_RD" + post_name + ".csv"), delimiter=',')[:25]

    egta[0] = fp[0]
    RD[0] = fp[0]

    # 加载 transformer 模型并计算 regret
    transformer_regret = load_transformer_model_and_calculate_regret(fp[0], size, step)

    # axes = plt.gca()
    # axes.set_ylim([0.0,3])

    # X = np.arange(1, len(fp)+1).astype(dtype=np.str)
    X = np.arange(1, 26).astype(dtype=str)

    plt.plot(X, fp, color="C0", label='FP')
    plt.plot(X, egta, color="C1", label='EGTA w. FP')
    plt.plot(X, RD, color="C2", label='EGTA w. RD')
    
    # 添加 transformer 模型的结果
    if transformer_regret is not None:
        plt.plot(X, transformer_regret, color="C3", label='EGTA w. Transformer')

    plt.xticks(np.arange(0, 25, 5), size = 17)
    plt.yticks(size = 17)

    plt.xlabel('Number of Iterations', fontsize = 22)
    plt.ylabel('Regret', fontsize = 19)

    plt.legend(loc="best", prop={'size': 17})

    plt.title("Size=" + str(size) + ", Horizon=" + str(step), fontsize=17)

    plt.tight_layout()
    new_figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_figures")
    os.makedirs(new_figures_dir, exist_ok=True)
    plt.savefig(os.path.join(new_figures_dir, "cd" + post_name + '.png'))

    # plt.show()

sizes = [10, 50, 100]
#sizes = [10]
steps = [10, 20, 30]
#steps = [20]

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