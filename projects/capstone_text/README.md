# Capstone Text Classification

This mini-project compares several text-classification baselines on a small synthetic movie-review dataset.

## Models Compared

| Model | Type | Notes |
| --- | --- | --- |
| Unigram-LR | sklearn | logistic regression on bag-of-words |
| Bigram-LR | sklearn | logistic regression on bigrams |
| LSTM-Small | PyTorch (`src.models`) | `embed_dim=16`, `hidden_dim=32` |
| LSTM-Medium | PyTorch (`src.models`) | `embed_dim=32`, `hidden_dim=64` |

## Run

```bash
cd projects/capstone_text
python train.py
```

Results are written to `artifacts/`:

- `config_used.json`: exact config used for the run
- `results.json`: sorted leaderboard with confusion matrices
- `histories.json`: per-epoch train and validation metrics for LSTM models

## Edit Experiments

Change `config.json` to add new LSTM variants or tweak hyperparameters, then run `train.py` again.
