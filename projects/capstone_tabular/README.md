# Capstone Tabular Project

This directory turns the core workflow from `05_01_capstone_tabular_classification.ipynb` into a reproducible mini-project.

## Goal

- run a tabular binary classification project
- compare a classical baseline with an `MLP`
- persist experiment results

## How to Run

Run from the repository root:

```bash
/home/cliao/miniconda3/envs/olinml/bin/python projects/capstone_tabular/train.py
```

## Outputs

Running the script writes:

- `projects/capstone_tabular/artifacts/config_used.json`
- `projects/capstone_tabular/artifacts/results.json`
- `projects/capstone_tabular/artifacts/histories.json`

## Current Scope

- dataset: `sklearn.datasets.load_breast_cancer`
- baseline: `LogisticRegression`
- improved models: two `TabularMLP` variants
- metrics: validation accuracy, test accuracy, classification report, and confusion matrix

## Next Step

If you want to grow this into a more complete project, add:

- a cleaner split between `train.py`, `evaluate.py`, and `infer.py`
- exported plots
- a script that aggregates repeated runs
- a clearer conclusions section in the `README`
