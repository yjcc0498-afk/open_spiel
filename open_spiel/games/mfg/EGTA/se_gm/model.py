import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from open_spiel.games.mfg.EGTA.model_learning.strategy_feature_extractor import (
    StrategyFeatureExtractor,
)


class TF_Regressor(object):
    # 新增：统一封装两种输入编码路径。
    # 1. one_hot：保留原论文/原代码的基线做法
    # 2. transformer_stats：使用 [T, 6] 时序特征 + Transformer 编码策略
    # 这样训练和推理接口不需要在外层分叉太多逻辑，便于直接做基线对比。
    def __init__(self,
                 nn_params,
                 verbose='0',
                 checkpoint_dir=None,
                 only_new=False,
                 encoding='one_hot'):

        self._X_train = None
        self._Y_train = None
        self._new_X = None
        self._new_Y = None

        self._verbose = verbose
        self._checkpoint_dir = checkpoint_dir
        self._only_new = only_new
        self._encoding = encoding

        self._learning_rate = nn_params['learning_rate']
        self._epochs = nn_params['epoch']
        self._batch_size = nn_params['batch_size']
        self._output_sizes = nn_params['output_sizes']

        self._num_policies = nn_params.get('num_policies')
        self._sequence_length = nn_params.get('sequence_length')
        self._feature_dim = nn_params.get('feature_dim', 6)
        self._d_model = nn_params.get('d_model', 128)
        self._nhead = nn_params.get('nhead', 4)
        self._num_layers = nn_params.get('num_layers', 2)

        self._policy_features = None
        self._feature_extractor = None

        self._model = self._build_model()

    @property
    def encoding(self):
        return self._encoding

    @property
    def num_policies(self):
        return self._num_policies

    def _build_model(self):
        if self._encoding == 'transformer_stats':
            return self._build_transformer_model()
        return self._build_mlp_model()

    def _build_mlp_model(self):
        # 旧路径：one-hot(strategy) + mixed_weights -> MLP -> utility
        model = keras.Sequential(name="utility_model")
        for idx, size in enumerate(self._output_sizes[:-1]):
            model.add(layers.Dense(size, activation="relu", name="layer" + str(idx + 1)))
        model.add(layers.Dense(self._output_sizes[-1], name="output"))
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate))
        return model

    def _build_transformer_model(self):
        # 新路径：strategy_features[T, 6] -> Transformer Encoder -> pooling -> embedding
        # 再与 mixed strategy 拼接后送入 MLP，输出 utility。
        if self._num_policies is None or self._sequence_length is None:
            raise ValueError("Transformer mode requires num_policies and sequence_length.")

        strategy_input = keras.Input(
            shape=(self._sequence_length, self._feature_dim),
            name="strategy_features")
        mixed_input = keras.Input(shape=(self._num_policies,), name="mixed_weights")

        x = layers.Dense(self._d_model, name="feature_projection")(strategy_input)

        positions = np.arange(self._sequence_length)
        position_embedding = layers.Embedding(
            input_dim=self._sequence_length,
            output_dim=self._d_model,
            name="position_embedding")(positions)
        x = x + position_embedding

        for idx in range(self._num_layers):
            attn_output = layers.MultiHeadAttention(
                num_heads=self._nhead,
                key_dim=max(self._d_model // self._nhead, 1),
                name="mha_{}".format(idx))(x, x)
            x = layers.LayerNormalization(epsilon=1e-6, name="attn_ln_{}".format(idx))(x + attn_output)

            ff = layers.Dense(self._d_model * 2, activation="relu", name="ff_{}_1".format(idx))(x)
            ff = layers.Dense(self._d_model, name="ff_{}_2".format(idx))(ff)
            x = layers.LayerNormalization(epsilon=1e-6, name="ff_ln_{}".format(idx))(x + ff)

        pooled = layers.GlobalAveragePooling1D(name="sequence_pooling")(x)
        merged = layers.Concatenate(name="utility_input")([pooled, mixed_input])

        hidden = merged
        for idx, size in enumerate(self._output_sizes[:-1]):
            hidden = layers.Dense(size, activation="relu", name="mlp_{}".format(idx + 1))(hidden)
        output = layers.Dense(self._output_sizes[-1], name="output")(hidden)

        model = keras.Model(
            inputs={"strategy_features": strategy_input, "mixed_weights": mixed_input},
            outputs=output,
            name="transformer_utility_model")
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate))
        return model

    def refresh_policy_features(self, game, policies, force_recompute=False):
        # 为当前 empirical game 的所有 pure policy 预计算时序特征。
        # 内层求解时只做查表和前向推理，不在 predict 阶段重复模拟 30 步。
        if self._encoding != 'transformer_stats':
            return
        if self._policy_features is not None and not force_recompute:
            if self._policy_features.shape[0] == len(policies):
                return
        if self._feature_extractor is None:
            self._feature_extractor = StrategyFeatureExtractor(
                game=game,
                sequence_length=self._sequence_length)
        self._policy_features = self._feature_extractor.batch_extract(policies)

    def get_policy_feature(self, policy_idx):
        if self._encoding != 'transformer_stats':
            raise ValueError("Policy features are only available in transformer_stats mode.")
        if self._policy_features is None:
            raise ValueError("Policy features have not been initialized.")
        return self._policy_features[policy_idx]

    def build_sample_input(self, policy_idx, mixed_weights):
        # 统一构造单条训练/推理样本。
        # 外层不需要知道当前是 one-hot 还是 Transformer，只要通过这里取输入即可。
        if self._encoding == 'transformer_stats':
            if self._policy_features is None:
                raise ValueError("Policy features have not been initialized.")
            return {
                "strategy_features": np.asarray(self._policy_features[policy_idx], dtype=np.float32),
                "mixed_weights": np.asarray(mixed_weights, dtype=np.float32),
            }

        one_hot_policy = np.zeros(self._num_policies, dtype=np.float32)
        one_hot_policy[policy_idx] = 1.0
        return np.append(one_hot_policy, np.asarray(mixed_weights, dtype=np.float32))

    def fit(self):
        # 训练入口保持不变，但底层根据 encoding 选择不同的输入结构。
        if self._X_train is None or self._Y_train is None:
            raise ValueError("None data set.")

        if self._only_new:
            train_X = self._new_X
            train_Y = self._new_Y
        else:
            train_X = self._X_train
            train_Y = self._Y_train

        train_X = self._prepare_dataset_inputs(train_X)
        train_Y = np.asarray(train_Y)

        self._fit_history = self._model.fit(
            train_X,
            train_Y,
            batch_size=self._batch_size,
            epochs=self._epochs,
            verbose=self._verbose)

        return self._fit_history.history['loss'][-1]

    def update_dataset(self, train_X=None, train_Y=None, new_X=None, new_Y=None):
        if train_X is not None:
            self._X_train = self._to_storage(train_X)
        if train_Y is not None:
            self._Y_train = np.asarray(train_Y)
        if new_X is not None:
            self._new_X = self._to_storage(new_X)
        if new_Y is not None:
            self._new_Y = np.asarray(new_Y)

    def combine_new_data(self, new_X, new_Y):
        new_X = self._to_storage(new_X)
        new_Y = np.asarray(new_Y)
        if self._new_X is None:
            self._new_X = new_X
            self._new_Y = new_Y
            return
        self._new_X = self._concat_inputs(self._new_X, new_X)
        self._new_Y = np.concatenate((self._new_Y, new_Y), axis=0)

    def combine_new_old_data(self):
        if self._new_X is None or self._new_Y is None:
            return
        if self._X_train is None:
            self._X_train = self._copy_inputs(self._new_X)
            self._Y_train = np.copy(self._new_Y)
        else:
            self._X_train = self._concat_inputs(self._X_train, self._new_X)
            self._Y_train = np.concatenate((self._Y_train, self._new_Y), axis=0)
        self._new_X = None
        self._new_Y = None

    def predict(self, X):
        prepared_X = self._prepare_prediction_inputs(X)
        return self._model.predict(prepared_X, verbose=0)

    def save_model(self):
        if self._checkpoint_dir is None:
            return
        self._model.save(self._checkpoint_dir + '/tf_model.h5')
        if self._encoding == 'transformer_stats' and self._policy_features is not None:
            np.save(self._checkpoint_dir + '/strategy_features.npy', self._policy_features)

    def _prepare_prediction_inputs(self, X):
        if self._encoding == 'transformer_stats':
            if isinstance(X, dict):
                return {
                    "strategy_features": np.asarray(X["strategy_features"], dtype=np.float32),
                    "mixed_weights": np.asarray(X["mixed_weights"], dtype=np.float32),
                }
            raise ValueError("Transformer mode expects a dict of model inputs.")
        return np.asarray(X, dtype=np.float32)

    def _prepare_dataset_inputs(self, X):
        if self._encoding == 'transformer_stats':
            return {
                "strategy_features": np.asarray(X["strategy_features"], dtype=np.float32),
                "mixed_weights": np.asarray(X["mixed_weights"], dtype=np.float32),
            }
        return np.asarray(X, dtype=np.float32)

    def _to_storage(self, X):
        # Transformer 模式下，数据不再是单个扁平向量，而是双输入：
        # strategy_features 和 mixed_weights。
        # 这里统一把外层传入的样本整理成模型可训练的存储格式。
        if self._encoding == 'transformer_stats':
            if isinstance(X, dict):
                return {
                    "strategy_features": np.asarray(X["strategy_features"], dtype=np.float32),
                    "mixed_weights": np.asarray(X["mixed_weights"], dtype=np.float32),
                }
            if isinstance(X, list):
                return {
                    "strategy_features": np.asarray(
                        [sample["strategy_features"] for sample in X], dtype=np.float32),
                    "mixed_weights": np.asarray(
                        [sample["mixed_weights"] for sample in X], dtype=np.float32),
                }
            raise ValueError("Transformer mode expects dict inputs.")
        return np.asarray(X, dtype=np.float32)

    def _copy_inputs(self, X):
        if self._encoding == 'transformer_stats':
            return {
                "strategy_features": np.copy(X["strategy_features"]),
                "mixed_weights": np.copy(X["mixed_weights"]),
            }
        return np.copy(X)

    def _concat_inputs(self, X1, X2):
        if self._encoding == 'transformer_stats':
            return {
                "strategy_features": np.concatenate(
                    (X1["strategy_features"], X2["strategy_features"]), axis=0),
                "mixed_weights": np.concatenate(
                    (X1["mixed_weights"], X2["mixed_weights"]), axis=0),
            }
        return np.concatenate((X1, X2), axis=0)
