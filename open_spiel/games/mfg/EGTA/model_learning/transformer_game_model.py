import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def _init_xavier(module):
    for param in module.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)


def build_mlp(input_dim, hidden_dim=256, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class _BaseTransformerEncoder(nn.Module):
    """Shared Transformer encoder that returns one sequence embedding."""

    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        _init_xavier(self)

    def _encode_sequence(self, tokens):
        batch_size = tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, tokens], dim=1)
        sequence = self.pos_encoder(sequence)
        encoded = self.transformer_encoder(sequence)
        return encoded[:, 0, :]


class PaperMLPGameModel(nn.Module):
    """Paper-style MLP regressor for [pure strategy one-hot, mixed strategy]."""

    def __init__(self, num_strategies, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_strategies = num_strategies
        self.output_layer = build_mlp(num_strategies * 2, hidden_dim, dropout)
        _init_xavier(self)

    def forward(self, strategy_one_hot, mixture):
        x = torch.cat([strategy_one_hot, mixture], dim=1)
        return self.output_layer(x)


class TransformerGameModel(_BaseTransformerEncoder):
    """Legacy one-hot strategy + mixed-weight Transformer regressor."""

    def __init__(self, num_strategies, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.num_strategies = num_strategies
        self.strategy_embedding = nn.Linear(num_strategies, d_model)
        self.mixture_embedding = nn.Linear(num_strategies, d_model)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )
        _init_xavier(self)

    def forward(self, strategy_one_hot, mixture):
        strategy_token = self.strategy_embedding(strategy_one_hot)
        mixture_token = self.mixture_embedding(mixture)
        tokens = torch.stack([strategy_token, mixture_token], dim=1)
        cls_output = self._encode_sequence(tokens)
        return self.output_layer(cls_output)


class TransformerStatsGameModel(_BaseTransformerEncoder):
    """Policy-sequence Transformer with a weighted mixture-policy embedding."""

    def __init__(self, mixture_dim, feature_dim=6, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, policy_features=None):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.feature_dim = feature_dim
        self.mixture_dim = mixture_dim
        self.step_feature_projection = nn.Linear(feature_dim, d_model)

        if policy_features is None:
            self.register_buffer("policy_features", None)
            mlp_input_dim = d_model + mixture_dim
        else:
            self.register_buffer("policy_features", torch.tensor(policy_features, dtype=torch.float32))
            # z_i, z_sigma, z_i * z_sigma, |z_i - z_sigma|, and raw sigma.
            mlp_input_dim = d_model * 4 + mixture_dim

        self.output_layer = build_mlp(mlp_input_dim, dim_feedforward, dropout)
        _init_xavier(self)

    def encode_strategy(self, strategy_features):
        step_tokens = self.step_feature_projection(strategy_features)
        return self._encode_sequence(step_tokens)

    def encode_policy_bank(self):
        return self.encode_strategy(self.policy_features)

    def forward(self, strategy_features, mixture):
        strategy_embedding = self.encode_strategy(strategy_features)

        if self.policy_features is None:
            x = torch.cat([strategy_embedding, mixture], dim=1)
            return self.output_layer(x)

        policy_embeddings = self.encode_policy_bank()
        mixture_embedding = mixture @ policy_embeddings
        interaction = strategy_embedding * mixture_embedding
        distance = torch.abs(strategy_embedding - mixture_embedding)
        x = torch.cat(
            [strategy_embedding, mixture_embedding, interaction, distance, mixture],
            dim=1,
        )
        return self.output_layer(x)
