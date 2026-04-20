# Capstone Text Classification

Binary sentiment classification (positive / negative) on a synthetic movie-review dataset.
Mirrors the structure of `../capstone_tabular/`.

## Models compared

| Model | Type | Notes |
|-------|------|-------|
| Unigram-LR | sklearn | Logistic regression on bag-of-words |
| Bigram-LR | sklearn | Logistic regression on bigrams |
| LSTM-Small | PyTorch (src.models) | embed=16, hidden=32 |
| LSTM-Medium | PyTorch (src.models) | embed=32, hidden=64 |

## Run

```bash
cd projects/capstone_text
python train.py
```

Results are written to `artifacts/`:
- `config_used.json` — exact config for this run
- `results.json` — sorted leaderboard with confusion matrices
- `histories.json` — per-epoch train/val metrics for LSTM models

## Edit experiments

Change `config.json` to add new LSTM variants or tweak hyperparameters,
then re-run `train.py`. Results are always overwritten in `artifacts/`.
