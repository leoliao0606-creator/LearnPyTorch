"""Reproducible tabular capstone training script."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import TabularMLP
from src.tabular_data import load_breast_cancer_splits, make_tabular_loaders
from src.training import collect_predictions, run_classification_epoch
from src.utils import ensure_dir, resolve_device, save_json, set_seed


def train_mlp(config, split_data, loaders, seed: int, device: torch.device, use_amp: bool = False):
    set_seed(seed)
    amp_enabled = bool(use_amp and device.type == "cuda")
    model = TabularMLP(
        in_dim=split_data["x_train"].shape[1],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = run_classification_epoch(
            model,
            loaders["train"],
            loss_fn,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            grad_scaler=scaler,
        )
        val_loss, val_acc = run_classification_epoch(model, loaders["val"], loss_fn, optimizer=None, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def main():
    project_dir = Path(__file__).resolve().parent
    artifacts_dir = ensure_dir(project_dir / "artifacts")
    config_path = project_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    device = resolve_device(config.get("device", "auto"))
    use_amp = config.get("use_amp", False)
    set_seed(config["seed"])
    split_data = load_breast_cancer_splits(seed=config["seed"])
    baseline_cfg = config["baseline"]
    baseline_model = LogisticRegression(
        max_iter=baseline_cfg["max_iter"],
        random_state=config["seed"],
    )
    baseline_model.fit(split_data["x_train"], split_data["y_train"])
    baseline_val_preds = baseline_model.predict(split_data["x_val"])
    baseline_test_preds = baseline_model.predict(split_data["x_test"])

    results = [
        {
            "model": "LogisticRegression",
            "family": "baseline",
            "val_acc": round(accuracy_score(split_data["y_val"], baseline_val_preds), 6),
            "test_acc": round(accuracy_score(split_data["y_test"], baseline_test_preds), 6),
            "confusion_matrix": confusion_matrix(split_data["y_test"], baseline_test_preds).tolist(),
            "classification_report": classification_report(
                split_data["y_test"],
                baseline_test_preds,
                target_names=split_data["target_names"],
                output_dict=True,
                zero_division=0,
            ),
        }
    ]
    histories = {}

    for experiment in config["experiments"]:
        loaders = make_tabular_loaders(
            split_data,
            train_batch_size=config["train_batch_size"],
            eval_batch_size=config["eval_batch_size"],
            seed=config["seed"],
        )
        model, history = train_mlp(
            experiment,
            split_data,
            loaders,
            seed=config["seed"],
            device=device,
            use_amp=use_amp,
        )
        val_preds, val_targets = collect_predictions(model, loaders["val"], device=device)
        test_preds, test_targets = collect_predictions(model, loaders["test"], device=device)
        histories[experiment["name"]] = history
        results.append(
            {
                "model": experiment["name"],
                "family": "torch",
                "config": experiment,
                "val_acc": round(accuracy_score(val_targets.numpy(), val_preds.numpy()), 6),
                "test_acc": round(accuracy_score(test_targets.numpy(), test_preds.numpy()), 6),
                "confusion_matrix": confusion_matrix(test_targets.numpy(), test_preds.numpy()).tolist(),
                "classification_report": classification_report(
                    test_targets.numpy(),
                    test_preds.numpy(),
                    target_names=split_data["target_names"],
                    output_dict=True,
                    zero_division=0,
                ),
            }
        )

    sorted_results = sorted(results, key=lambda row: (row["test_acc"], row["val_acc"]), reverse=True)

    config_used = {
        **config,
        "runtime": {
            "device": str(device),
            "amp_enabled": bool(use_amp and device.type == "cuda"),
        },
    }
    save_json(config_used, artifacts_dir / "config_used.json")
    save_json(histories, artifacts_dir / "histories.json")
    save_json(sorted_results, artifacts_dir / "results.json")

    print("Saved artifacts:")
    print("-", artifacts_dir / "config_used.json")
    print("-", artifacts_dir / "histories.json")
    print("-", artifacts_dir / "results.json")
    print()
    print("Leaderboard:")
    for row in sorted_results:
        print(
            f"{row['model']}: val_acc={row['val_acc']:.4f}, "
            f"test_acc={row['test_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
