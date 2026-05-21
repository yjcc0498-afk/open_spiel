import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from .transformer_game_model import PaperMLPGameModel, TransformerStatsGameModel
except ImportError:
    from transformer_game_model import PaperMLPGameModel, TransformerStatsGameModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EGTA_DIR = os.path.dirname(SCRIPT_DIR)
EPS = 1e-8


class OneHotMFGDataset(Dataset):
    """Dataset for the original paper-style one-hot utility inputs."""

    def __init__(self, strategies, mixtures, utilities):
        self.strategies = torch.tensor(strategies, dtype=torch.float32)
        self.mixtures = torch.tensor(mixtures, dtype=torch.float32)
        self.utilities = torch.tensor(utilities, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.utilities)

    def __getitem__(self, idx):
        return self.strategies[idx], self.mixtures[idx], self.utilities[idx]


class TransformerStatsDataset(Dataset):
    """Dataset for sequence strategy features + mixed weights."""

    def __init__(self, strategy_features, mixtures, utilities):
        self.strategy_features = torch.tensor(strategy_features, dtype=torch.float32)
        self.mixtures = torch.tensor(mixtures, dtype=torch.float32)
        self.utilities = torch.tensor(utilities, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.utilities)

    def __getitem__(self, idx):
        return self.strategy_features[idx], self.mixtures[idx], self.utilities[idx]


def calculate_r2(predictions, targets):
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def calculate_mse(predictions, targets):
    return torch.mean((predictions - targets) ** 2).item()


def random_train_val_split(*arrays, test_size=0.2, random_state=42):
    """Simple sklearn-free replacement for train_test_split."""
    if not arrays:
        raise ValueError("At least one array is required.")
    num_samples = len(arrays[0])
    for array in arrays[1:]:
        if len(array) != num_samples:
            raise ValueError("All arrays must have the same number of samples.")

    rng = np.random.RandomState(random_state)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(round(num_samples * test_size))
    val_size = min(max(val_size, 1), num_samples - 1) if num_samples > 1 else num_samples
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    split_arrays = []
    for array in arrays:
        array = np.asarray(array)
        split_arrays.extend([array[train_indices], array[val_indices]])
    return tuple(split_arrays)


def split_train_val(arrays, args, policy_indices=None):
    if args.split_mode == "random":
        return random_train_val_split(*arrays, test_size=args.val_split, random_state=42), None

    if policy_indices is None:
        raise ValueError("strategy_holdout split requires policy indices.")

    policy_indices = np.asarray(policy_indices)
    unique_policies = np.unique(policy_indices)
    if len(unique_policies) < 2:
        raise ValueError("Need at least two policies for strategy_holdout split.")

    holdout_count = min(max(args.holdout_strategy_count, 1), len(unique_policies) - 1)
    rng = np.random.RandomState(42)
    holdout_policies = np.sort(rng.choice(unique_policies, size=holdout_count, replace=False))
    val_mask = np.isin(policy_indices, holdout_policies)
    train_mask = ~val_mask

    split_arrays = []
    for array in arrays:
        array = np.asarray(array)
        split_arrays.extend([array[train_mask], array[val_mask]])
    return tuple(split_arrays), holdout_policies.tolist()


def infer_block_policy_indices(num_rows, num_policies):
    if num_policies <= 0 or num_rows % num_policies != 0:
        return None
    samples_per_policy = num_rows // num_policies
    return np.repeat(np.arange(num_policies), samples_per_policy)


def _standardize_targets(train_utilities, val_utilities):
    y_mean = np.mean(train_utilities, dtype=np.float64)
    y_std = np.std(train_utilities, dtype=np.float64)
    if y_std < EPS:
        y_std = 1.0
    return (
        (train_utilities - y_mean) / y_std,
        (val_utilities - y_mean) / y_std,
        float(y_mean),
        float(y_std),
    )


def _standardize_strategy_features(train_features, val_features, policy_features):
    feature_mean = np.mean(train_features, axis=(0, 1), keepdims=True)
    feature_std = np.std(train_features, axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < EPS, 1.0, feature_std)

    train_features = (train_features - feature_mean) / feature_std
    val_features = (val_features - feature_mean) / feature_std
    if policy_features is not None:
        policy_features = (policy_features - feature_mean) / feature_std
    return train_features, val_features, policy_features, feature_mean, feature_std


def _denormalize(tensor, y_mean, y_std):
    return tensor * y_std + y_mean


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, y_mean=0.0,
                y_std=1.0, patience=30, verbose=False):
    """Train on normalized targets while reporting metrics on the raw utility scale."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, patience // 3)
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_r2 = -float("inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_objective_loss = 0.0
        train_predictions = []
        train_targets = []

        for first_input, second_input, utility in train_loader:
            first_input = first_input.to(device)
            second_input = second_input.to(device)
            utility = utility.to(device)

            optimizer.zero_grad()
            output = model(first_input, second_input)
            loss = criterion(output, utility)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_objective_loss += loss.item()
            train_predictions.append(_denormalize(output.detach().cpu(), y_mean, y_std))
            train_targets.append(_denormalize(utility.detach().cpu(), y_mean, y_std))

        train_predictions = torch.cat(train_predictions) if train_predictions else torch.empty(0, 1)
        train_targets = torch.cat(train_targets) if train_targets else torch.empty(0, 1)
        train_mse = calculate_mse(train_predictions, train_targets) if len(train_predictions) else 0.0
        train_r2 = calculate_r2(train_predictions, train_targets) if len(train_predictions) else 0.0
        train_losses.append(train_mse)
        train_r2_scores.append(train_r2)

        model.eval()
        val_objective_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for first_input, second_input, utility in val_loader:
                first_input = first_input.to(device)
                second_input = second_input.to(device)
                utility = utility.to(device)

                output = model(first_input, second_input)
                loss = criterion(output, utility)
                val_objective_loss += loss.item()
                val_predictions.append(_denormalize(output.cpu(), y_mean, y_std))
                val_targets.append(_denormalize(utility.cpu(), y_mean, y_std))

        avg_val_objective_loss = val_objective_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_objective_loss)

        val_predictions = torch.cat(val_predictions) if val_predictions else torch.empty(0, 1)
        val_targets = torch.cat(val_targets) if val_targets else torch.empty(0, 1)
        val_mse = calculate_mse(val_predictions, val_targets) if len(val_predictions) else 0.0
        val_r2 = calculate_r2(val_predictions, val_targets) if len(val_predictions) else 0.0
        val_losses.append(val_mse)
        val_r2_scores.append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_loss = val_mse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                "Epoch {}/{} | Train MSE {:.6f} | Val MSE {:.6f} | Train R2 {:.6f} | Val R2 {:.6f} | LR {:.2e}".format(
                    epoch + 1, epochs, train_mse, val_mse, train_r2, val_r2, current_lr
                )
            )

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print("Early stopping at epoch {}. Best epoch: {}".format(epoch + 1, best_epoch))
            break

    if verbose:
        print("\nFinal Metrics")
        print("Train MSE: {:.6f}".format(train_losses[-1]))
        print("Val MSE: {:.6f}".format(val_losses[-1]))
        print("Train R2: {:.6f}".format(train_r2_scores[-1]))
        print("Val R2: {:.6f}".format(val_r2_scores[-1]))
        print("Best Val MSE: {:.6f}".format(best_val_loss))
        print("Best Val R2: {:.6f} at epoch {}".format(best_val_r2, best_epoch))

    return train_losses, val_losses, train_r2_scores, val_r2_scores, best_state, best_epoch, best_val_loss, best_val_r2


def _default_data_dir():
    return os.path.join(EGTA_DIR, "root_result", "mfg_crowd_modelling_it_10_grid_True_sample_100_gendata_2026-04-06_20-29-38")


def load_one_hot_data(data_dir):
    x_file = os.path.join(data_dir, "utility_X.csv")
    y_file = os.path.join(data_dir, "utility_Y.csv")
    print("Loading one-hot data from:", x_file)
    print("Target file:", y_file)

    X = np.loadtxt(x_file, delimiter=",", dtype=np.float32)
    Y = np.loadtxt(y_file, delimiter=",", dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if np.ndim(Y) == 0:
        Y = np.asarray([Y], dtype=np.float32)

    _, total_features = X.shape
    num_strategies = total_features // 2
    strategies = X[:, :num_strategies]
    mixtures = X[:, num_strategies:]
    utilities = np.asarray(Y, dtype=np.float32).reshape(-1, 1)
    return strategies, mixtures, utilities, num_strategies


def load_transformer_stats_data(data_dir):
    feature_file = os.path.join(data_dir, "utility_strategy_features.npy")
    mixture_file = os.path.join(data_dir, "utility_mixed_weights.npy")
    policy_feature_file = os.path.join(data_dir, "policy_features.npy")
    y_file = os.path.join(data_dir, "utility_Y.csv")
    print("Loading transformer_stats data from:", feature_file)
    print("Mixed weights file:", mixture_file)
    print("Policy feature bank:", policy_feature_file)
    print("Target file:", y_file)

    strategy_features = np.load(feature_file).astype(np.float32)
    mixtures = np.load(mixture_file).astype(np.float32)
    policy_features = None
    if os.path.exists(policy_feature_file):
        policy_features = np.load(policy_feature_file).astype(np.float32)
    utilities = np.loadtxt(y_file, delimiter=",", dtype=np.float32)

    if strategy_features.ndim == 2:
        strategy_features = strategy_features.reshape(1, *strategy_features.shape)
    if mixtures.ndim == 1:
        mixtures = mixtures.reshape(1, -1)
    if policy_features is not None and policy_features.ndim == 2:
        policy_features = policy_features.reshape(1, *policy_features.shape)
    if np.ndim(utilities) == 0:
        utilities = np.asarray([utilities], dtype=np.float32)

    utilities = np.asarray(utilities, dtype=np.float32).reshape(-1, 1)
    return strategy_features, mixtures, utilities, policy_features


def build_dataloaders(args):
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else _default_data_dir()
    print("Data dir:", data_dir)
    print("Encoding:", args.encoding)

    scalers = {}
    if args.encoding == "one_hot":
        strategies, mixtures, utilities, num_strategies = load_one_hot_data(data_dir)
        policy_indices = np.argmax(strategies, axis=1)
        split, holdout_policies = split_train_val(
            (strategies, mixtures, utilities), args, policy_indices=policy_indices
        )
        train_strategies, val_strategies, train_mixtures, val_mixtures, train_utilities, val_utilities = split
        train_utilities, val_utilities, y_mean, y_std = _standardize_targets(train_utilities, val_utilities)

        train_dataset = OneHotMFGDataset(train_strategies, train_mixtures, train_utilities)
        val_dataset = OneHotMFGDataset(val_strategies, val_mixtures, val_utilities)
        model = PaperMLPGameModel(
            num_strategies=num_strategies,
            hidden_dim=args.dim_feedforward,
            dropout=args.dropout,
        )
        metadata = {
            "num_strategies": num_strategies,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "y_mean": y_mean,
            "y_std": y_std,
            "uses_policy_feature_bank": False,
            "split_mode": args.split_mode,
            "holdout_policies": holdout_policies,
        }
    else:
        strategy_features, mixtures, utilities, policy_features = load_transformer_stats_data(data_dir)
        num_policies = policy_features.shape[0] if policy_features is not None else 0
        policy_indices = infer_block_policy_indices(len(strategy_features), num_policies)
        split, holdout_policies = split_train_val(
            (strategy_features, mixtures, utilities), args, policy_indices=policy_indices
        )
        train_features, val_features, train_mixtures, val_mixtures, train_utilities, val_utilities = split
        train_features, val_features, policy_features, feature_mean, feature_std = _standardize_strategy_features(
            train_features, val_features, policy_features
        )
        train_utilities, val_utilities, y_mean, y_std = _standardize_targets(train_utilities, val_utilities)

        train_dataset = TransformerStatsDataset(train_features, train_mixtures, train_utilities)
        val_dataset = TransformerStatsDataset(val_features, val_mixtures, val_utilities)
        model = TransformerStatsGameModel(
            mixture_dim=train_mixtures.shape[1],
            feature_dim=train_features.shape[2],
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            policy_features=policy_features,
        )
        scalers["feature_mean"] = feature_mean.astype(np.float32)
        scalers["feature_std"] = feature_std.astype(np.float32)
        metadata = {
            "feature_shape": train_features.shape[1:],
            "mixture_dim": train_mixtures.shape[1],
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "y_mean": y_mean,
            "y_std": y_std,
            "uses_policy_feature_bank": policy_features is not None,
            "split_mode": args.split_mode,
            "holdout_policies": holdout_policies,
        }

    scalers["y_mean"] = np.asarray([metadata["y_mean"]], dtype=np.float32)
    scalers["y_std"] = np.asarray([metadata["y_std"]], dtype=np.float32)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    return data_dir, model, train_loader, val_loader, metadata, scalers


def save_training_plot(train_losses, val_losses, train_r2_scores, val_r2_scores, output_prefix):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Raw-scale MSE")
    plt.title("Training and Validation MSE")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_r2_scores, label="Train R2")
    plt.plot(val_r2_scores, label="Validation R2")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Training and Validation R2 Score")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(train_losses, train_r2_scores, label="Train")
    plt.scatter(val_losses, val_r2_scores, label="Validation")
    plt.xlabel("Raw-scale MSE")
    plt.ylabel("R2 Score")
    plt.title("MSE vs R2 Score")
    plt.legend()

    tail = min(10, len(train_losses))
    plt.subplot(2, 2, 4)
    epochs = list(range(len(train_losses) - tail + 1, len(train_losses) + 1))
    plt.plot(epochs, train_losses[-tail:], label="Train MSE")
    plt.plot(epochs, val_losses[-tail:], label="Validation MSE")
    plt.plot(epochs, train_r2_scores[-tail:], label="Train R2")
    plt.plot(epochs, val_r2_scores[-tail:], label="Validation R2")
    plt.xlabel("Epoch")
    plt.title("Last {} Epochs".format(tail))
    plt.legend()

    plt.tight_layout()
    plot_path = output_prefix + "_metrics.png"
    plt.savefig(plot_path)
    print("Saved training plot to:", plot_path)


def main():
    parser = argparse.ArgumentParser(description="Train utility model for MFG EGTA data.")
    parser.add_argument("--size", type=int, default=10, help="Game size metadata used in the saved model name.")
    parser.add_argument("--step", type=int, default=20, help="Game horizon metadata used in the saved model name.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing generated utility data.")
    parser.add_argument("--encoding", type=str, default="one_hot", choices=["one_hot", "transformer_stats"],
                        help="Select whether to train on CSV one-hot inputs or transformer_stats .npy inputs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden size.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer encoder layers.")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Feed-forward hidden size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience. Use 0 to disable.")
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "strategy_holdout"],
                        help="random matches the paper-style row split; strategy_holdout tests unseen pure policies.")
    parser.add_argument("--holdout_strategy_count", type=int, default=3,
                        help="Number of pure policies held out when split_mode=strategy_holdout.")
    parser.add_argument("--verbose", action="store_true", help="Print periodic training metrics.")
    args = parser.parse_args()

    data_dir, model, train_loader, val_loader, metadata, scalers = build_dataloaders(args)
    print("Train size:", metadata["train_size"])
    print("Val size:", metadata["val_size"])
    for key, value in metadata.items():
        if key not in ("train_size", "val_size"):
            print("{}: {}".format(key, value))

    train_losses, val_losses, train_r2_scores, val_r2_scores, best_state, best_epoch, best_val_loss, best_val_r2 = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        y_mean=metadata["y_mean"],
        y_std=metadata["y_std"],
        patience=args.patience,
        verbose=args.verbose,
    )

    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_prefix = "trained_model_{}_size{}_step{}".format(args.encoding, args.size, args.step)
    model_path = os.path.join(models_dir, model_prefix + ".pth")
    best_model_path = os.path.join(models_dir, model_prefix + "_best.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(best_state, best_model_path)
    print("Saved final model to:", model_path)
    print("Saved best model to:", best_model_path)

    scaler_path = os.path.join(models_dir, model_prefix + "_scalers.npz")
    np.savez(scaler_path, **scalers)
    print("Saved scalers to:", scaler_path)

    metrics_path = os.path.join(models_dir, model_prefix)
    save_training_plot(train_losses, val_losses, train_r2_scores, val_r2_scores, metrics_path)

    summary_path = os.path.join(models_dir, model_prefix + "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("data_dir={}\n".format(data_dir))
        summary_file.write("encoding={}\n".format(args.encoding))
        summary_file.write("train_size={}\n".format(metadata["train_size"]))
        summary_file.write("val_size={}\n".format(metadata["val_size"]))
        summary_file.write("final_train_loss={:.6f}\n".format(train_losses[-1]))
        summary_file.write("final_val_loss={:.6f}\n".format(val_losses[-1]))
        summary_file.write("final_train_r2={:.6f}\n".format(train_r2_scores[-1]))
        summary_file.write("final_val_r2={:.6f}\n".format(val_r2_scores[-1]))
        summary_file.write("best_epoch={}\n".format(best_epoch))
        summary_file.write("best_val_loss={:.6f}\n".format(best_val_loss))
        summary_file.write("best_val_r2={:.6f}\n".format(best_val_r2))
        summary_file.write("y_mean={:.8f}\n".format(metadata["y_mean"]))
        summary_file.write("y_std={:.8f}\n".format(metadata["y_std"]))
        summary_file.write("uses_policy_feature_bank={}\n".format(metadata["uses_policy_feature_bank"]))
        summary_file.write("split_mode={}\n".format(metadata["split_mode"]))
        summary_file.write("holdout_policies={}\n".format(metadata["holdout_policies"]))
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()
