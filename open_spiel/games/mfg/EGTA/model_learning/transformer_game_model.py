import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

class PositionalEncoding(nn.Module): 
    """为序列添加位置编码"""
    def __init__(self, d_model, max_len=10): 
        super().__init__() 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe.unsqueeze(0)) 
    
    def forward(self, x): 
        # x shape: (batch, seq_len, d_model) 
        return x + self.pe[:, :x.size(1)] 

class TransformerGameModel(nn.Module): 
    """用Transformer替代MLP的游戏模型学习器"""
    
    def __init__(self, num_strategies, d_model=128, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout=0.1): 
        """
        Args: 
            num_strategies: 策略数量 (r) 
            d_model: embedding维度 
            nhead: 注意力头数 
            num_layers: Transformer层数 
        """
        super().__init__() 
        
        self.num_strategies = num_strategies 
        self.d_model = d_model 
        
        # 1. 两个输入的Embedding层 
        # 策略索引的embedding (one-hot -> dense) 
        self.strategy_embedding = nn.Linear(num_strategies, d_model) 
        
        # 混合策略的embedding (σ向量 -> dense) 
        # 这里也可以用单独的线性层，或者共享参数 
        self.mixture_embedding = nn.Linear(num_strategies, d_model) 
        
        # 2. 位置编码 
        self.pos_encoder = PositionalEncoding(d_model) 
        
        # 3. Transformer Encoder层 
        encoder_layer = nn.TransformerEncoderLayer( 
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='relu', 
            batch_first=True  # 使用batch_first方便处理 
        ) 
        self.transformer_encoder = nn.TransformerEncoder( 
            encoder_layer, 
            num_layers=num_layers 
        ) 
        
        # 4. [CLS] token 
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) 
        
        # 5. 输出层 
        self.output_layer = nn.Sequential( 
            nn.Linear(d_model, dim_feedforward), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim_feedforward, 1) 
        ) 
        
        # 初始化参数 
        self._init_weights() 
    
    def _init_weights(self): 
        for p in self.parameters(): 
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p) 
    
    def forward(self, strategy_idx, mixture): 
        """
        Args: 
            strategy_idx: (batch, num_strategies) one-hot编码的策略索引 
            mixture: (batch, num_strategies) 混合策略σ向量 
        
        Returns: 
            utility: (batch, 1) 预测的效用值 
        """
        batch_size = strategy_idx.shape[0] 
        
        # 1. Embedding两个输入 
        # (batch, d_model) 
        strategy_emb = self.strategy_embedding(strategy_idx) 
        mixture_emb = self.mixture_embedding(mixture) 
        
        # 2. 添加[CLS] token 
        # 将[CLS] token扩展到batch维度 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model) 
        
        # 3. 构建序列: [CLS, 策略, 混合策略]
        sequence = torch.stack([strategy_emb, mixture_emb], dim=1)  # (batch, 2, d_model) 
        sequence = torch.cat([cls_tokens, sequence], dim=1)  # (batch, 3, d_model) 
        
        # 4. 添加位置编码 
        sequence = self.pos_encoder(sequence) 
        
        # 5. Transformer编码  
        encoded = self.transformer_encoder(sequence)  # (batch, 3, d_model) 
        
        # 6. 取[CLS] token的输出用于预测 
        cls_output = encoded[:, 0, :]  # (batch, d_model) 
        
        # 7. 输出层 
        utility = self.output_layer(cls_output)  # (batch, 1) 
        
        return utility 