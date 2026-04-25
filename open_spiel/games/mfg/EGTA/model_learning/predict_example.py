import torch
from transformer_game_model import TransformerGameModel

# 模型参数
num_strategies = 2
d_model = 128

# 加载模型
import os
model = TransformerGameModel(
    num_strategies=num_strategies,
    d_model=d_model,
    nhead=4,
    num_layers=2
)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained_model.pth")
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"模型加载成功: {model_path}")

# 预测函数
def predict_utility(strategy, mixture):
    """
    预测给定策略和混合策略的效用值
    
    Args:
        strategy: (num_strategies,) one-hot编码的策略
        mixture: (num_strategies,) 混合策略向量
        
    Returns:
        float: 预测的效用值
    """
    strategy_tensor = torch.tensor(strategy, dtype=torch.float32).unsqueeze(0)
    mixture_tensor = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(strategy_tensor, mixture_tensor)
        return prediction.item()

# 示例预测
sample_strategy = [1.0, 0.0]  
sample_mixture = [0.25, 0.75]  

predicted_value = predict_utility(sample_strategy, sample_mixture)
print(f"预测的效用值: {predicted_value}")