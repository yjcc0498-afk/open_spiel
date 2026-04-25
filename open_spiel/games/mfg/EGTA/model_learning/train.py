import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import matplotlib.pyplot as plt 
from transformer_game_model import TransformerGameModel
from sklearn.model_selection import train_test_split 

class MFGDataset(Dataset): 
    """MFG数据集"""
    def __init__(self, strategies, mixtures, utilities): 
        """
        Args: 
            strategies: (N, r) one-hot策略索引 
            mixtures: (N, r) 混合策略向量 
            utilities: (N, 1) 真实效用值 
        """
        self.strategies = torch.FloatTensor(strategies) 
        self.mixtures = torch.FloatTensor(mixtures) 
        self.utilities = torch.FloatTensor(utilities).reshape(-1, 1) 
    
    def __len__(self): 
        return len(self.utilities) 
    
    def __getitem__(self, idx): 
        return self.strategies[idx], self.mixtures[idx], self.utilities[idx] 

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, verbose=False): 
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = model.to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.MSELoss() 
    
    train_losses = [] 
    val_losses = [] 
    train_r2_scores = [] 
    val_r2_scores = [] 
    
    def calculate_r2(predictions, targets): 
        """计算R2决定系数"""
        ss_res = torch.sum((targets - predictions) ** 2) 
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2) 
        return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0 
    
    for epoch in range(epochs): 
        # 训练 
        model.train() 
        train_loss = 0 
        train_predictions = [] 
        train_targets = [] 
        
        for strategy, mixture, utility in train_loader: 
            strategy, mixture, utility = strategy.to(device), mixture.to(device), utility.to(device) 
            
            optimizer.zero_grad() 
            output = model(strategy, mixture) 
            loss = criterion(output, utility) 
            loss.backward() 
            optimizer.step() 
            
            train_loss += loss.item() 
            train_predictions.append(output.cpu()) 
            train_targets.append(utility.cpu()) 
        
        avg_train_loss = train_loss / len(train_loader) 
        train_losses.append(avg_train_loss) 
        
        # 计算训练集R2
        if train_predictions: 
            train_predictions = torch.cat(train_predictions) 
            train_targets = torch.cat(train_targets) 
            train_r2 = calculate_r2(train_predictions, train_targets) 
            train_r2_scores.append(train_r2) 
        else: 
            train_r2 = 0 
            train_r2_scores.append(train_r2) 
        
        # 验证 
        model.eval() 
        val_loss = 0 
        val_predictions = [] 
        val_targets = [] 
        
        with torch.no_grad(): 
            for strategy, mixture, utility in val_loader: 
                strategy, mixture, utility = strategy.to(device), mixture.to(device), utility.to(device) 
                output = model(strategy, mixture) 
                loss = criterion(output, utility) 
                val_loss += loss.item() 
                
                # 收集预测值和真实值用于计算R2 
                val_predictions.append(output.cpu()) 
                val_targets.append(utility.cpu()) 
        
        avg_val_loss = val_loss / len(val_loader) 
        val_losses.append(avg_val_loss) 
        
        # 计算验证集R2 
        if val_predictions: 
            val_predictions = torch.cat(val_predictions) 
            val_targets = torch.cat(val_targets) 
            val_r2 = calculate_r2(val_predictions, val_targets) 
            val_r2_scores.append(val_r2) 
        else: 
            val_r2 = 0 
            val_r2_scores.append(val_r2) 
        
        if verbose and (epoch + 1) % 10 == 0: 
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Train R2: {train_r2:.6f}, Val R2: {val_r2:.6f}') 
    
    # 最终评估
    if verbose:
        print("\n最终评估结果:")
        print(f"训练集 MSE: {avg_train_loss:.6f}")
        print(f"验证集 MSE: {avg_val_loss:.6f}")
        print(f"训练集 R2: {train_r2:.6f}")
        print(f"验证集 R2: {val_r2:.6f}")
    
    return train_losses, val_losses, train_r2_scores, val_r2_scores 

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='训练 Transformer 模型')
    parser.add_argument('--size', type=int, default=10, help='游戏大小')
    parser.add_argument('--step', type=int, default=20, help='游戏时间范围')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录路径')
    args = parser.parse_args()
    
    # 参数设置 
    batch_size = 32 
    d_model = 128 
    epochs = 100
    learning_rate = 0.001
    size = args.size
    step = args.step
    
    # 数据文件路径
    import os
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "root_result", "mfg_crowd_modelling_it_10_grid_True_sample_100_gendata_2026-04-06_20-29-38")
    x_file = os.path.join(data_dir, "utility_X.csv")
    y_file = os.path.join(data_dir, "utility_Y.csv")
    print(f"数据文件路径: {x_file}")
    print(f"数据文件存在: {os.path.exists(x_file)}")
    print(f"游戏大小: {size}")
    print(f"游戏时间范围: {step}")
    
    # 加载数据
    import numpy as np
    X = np.loadtxt(x_file, delimiter=',', dtype=float)
    Y = np.loadtxt(y_file, delimiter=',', dtype=float)
    
    # 解析数据：前N列是策略（one-hot），后N列是混合策略
    # 自动计算策略数量
    num_samples, total_features = X.shape
    num_strategies = total_features // 2
    
    # 分割数据
    strategies = X[:, :num_strategies]
    mixtures = X[:, num_strategies:]
    utilities = Y.reshape(-1, 1)
    
    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split
    train_strategies, val_strategies, train_mixtures, val_mixtures, train_utilities, val_utilities = train_test_split(
        strategies, mixtures, utilities, test_size=0.2, random_state=42
    )
    
    # 创建模型 
    model = TransformerGameModel( 
        num_strategies=num_strategies, 
        d_model=d_model, 
        nhead=4, 
        num_layers=2 
    )
    
    print(f"训练集大小: {len(train_strategies)}")
    print(f"验证集大小: {len(val_strategies)}")
    
    # 创建数据集和数据加载器
    train_dataset = MFGDataset(train_strategies, train_mixtures, train_utilities)
    val_dataset = MFGDataset(val_strategies, val_mixtures, val_utilities)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 训练
    train_losses, val_losses, train_r2_scores, val_r2_scores = train_model(
        model, train_loader, val_loader, 
        epochs=epochs, lr=learning_rate, verbose=True
    )
    
    # 保存模型
    import os
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'trained_model_size{size}_step{step}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    # 可视化训练结果
    # 绘制损失曲线
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制R2曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_r2_scores, label='Train R2')
    plt.plot(val_r2_scores, label='Validation R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('Training and Validation R2 Score')
    plt.legend()
    
    # 绘制训练集MSE和R2的关系
    plt.subplot(2, 2, 3)
    plt.scatter(train_losses, train_r2_scores, label='Train')
    plt.scatter(val_losses, val_r2_scores, label='Validation')
    plt.xlabel('MSE Loss')
    plt.ylabel('R2 Score')
    plt.title('MSE vs R2 Score')
    plt.legend()
    
    # 绘制最后10个epoch的详细信息
    plt.subplot(2, 2, 4)
    plt.plot(range(epochs-9, epochs+1), train_losses[-10:], label='Train Loss')
    plt.plot(range(epochs-9, epochs+1), val_losses[-10:], label='Validation Loss')
    plt.plot(range(epochs-9, epochs+1), train_r2_scores[-10:], label='Train R2')
    plt.plot(range(epochs-9, epochs+1), val_r2_scores[-10:], label='Validation R2')
    plt.xlabel('Epoch')
    plt.title('Last 10 Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()