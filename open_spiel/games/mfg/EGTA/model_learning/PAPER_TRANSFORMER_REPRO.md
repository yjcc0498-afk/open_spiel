# Paper Coarse-Coding Transformer Reproduction

This experiment keeps the paper's coarse-coded utility input fixed:

```text
[one_hot(pure_strategy), mixed_strategy_sigma] -> utility
```

The controlled change is only the neural regressor:

- `--model_type mlp`: paper-style two-hidden-layer MLP baseline.
- `--model_type transformer`: Transformer encoder over the same coarse-coded inputs.

Do not use `--model_type transformer_stats` for this reproduction. That path changes
the strategy representation and belongs to the future-work style experiment.

## 1. Generate Paper-Style Utility Data

The Table 4 presets set the game size, horizon, EGTA iterations, and sample target:

- `mean_field_lin_quad`: about 10 strategies, 3000 mixed-strategy samples.
- `mfg_crowd_modelling`: about 18 strategies, 6000 mixed-strategy samples.
- `mfg_crowd_modelling_2d`: about 18 strategies, 6000 mixed-strategy samples.

Example:

```bash
python open_spiel/games/mfg/EGTA/model_learning/generate_data.py \
  --game_name=mfg_crowd_modelling \
  --encoding=one_hot \
  --sampling_mode=hybrid \
  --root_result_folder=root_result
```

Repeat with:

```text
mean_field_lin_quad
mfg_crowd_modelling
mfg_crowd_modelling_2d
```

Use the same generated `data_dir` for both MLP and Transformer runs.

## 2. Train Paper MLP Baseline

```bash
python open_spiel/games/mfg/EGTA/model_learning/train.py \
  --data_dir=<generated_data_dir> \
  --encoding=one_hot \
  --model_type=mlp \
  --training_steps=1000 \
  --batch_size=32 \
  --lr=0.001 \
  --optimizer=adam \
  --weight_decay=0.0 \
  --patience=0 \
  --val_split=0.3 \
  --seed=42 \
  --size=100 \
  --step=30 \
  --verbose
```

## 3. Train Transformer With Same Coarse Coding

```bash
python open_spiel/games/mfg/EGTA/model_learning/train.py \
  --data_dir=<generated_data_dir> \
  --encoding=one_hot \
  --model_type=transformer \
  --training_steps=1000 \
  --batch_size=32 \
  --lr=0.001 \
  --optimizer=adam \
  --weight_decay=0.0 \
  --patience=0 \
  --val_split=0.3 \
  --seed=42 \
  --d_model=128 \
  --nhead=4 \
  --num_layers=2 \
  --dim_feedforward=256 \
  --dropout=0.1 \
  --size=100 \
  --step=30 \
  --verbose
```

## 4. Learned-Utility FP/RD Runs

For online learned-utility FP/RD using the existing Keras `se_gm` path:

```bash
python open_spiel/games/mfg/EGTA/se_gm/egta_example.py \
  --game_name=mfg_crowd_modelling \
  --meta_strategy_method=RD \
  --model_type=mlp
```

Transformer regressor with the same paper coarse coding:

```bash
python open_spiel/games/mfg/EGTA/se_gm/egta_example.py \
  --game_name=mfg_crowd_modelling \
  --meta_strategy_method=RD \
  --model_type=transformer
```

Run the same command pairs for LQ and 2D Crowd.

## 5. Report

Report the same quantities as the paper:

- Test R2 for LQ, 1D Crowd, and 2D Crowd.
- FP regret curves using true utility, MLP learned utility, and Transformer learned utility.
- RD regret curves using true utility, MLP learned utility, and Transformer learned utility.
- Mean-field distribution plots.
- Wasserstein distance by time step for 1D and 2D Crowd.
