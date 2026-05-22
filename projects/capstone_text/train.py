"""Reproducible text classification training script.

Mirrors the structure of projects/capstone_tabular/train.py.
Dataset: synthetic sentiment (same generator as 05_02 capstone notebook).
Models: Unigram-LR, Bigram-LR (sklearn), LSTM variants (PyTorch via src).
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import TextLSTMClassifier
from src.text_data import build_vocab, make_text_loaders
from src.training import collect_predictions, run_classification_epoch
from src.utils import ensure_dir, resolve_device, save_json, set_seed


# ---------------------------------------------------------------------------
# Dataset generation (same as 05_02 notebook)
# ---------------------------------------------------------------------------

_NOUNS = ["movie", "film", "story", "show", "plot", "episode"]
_POS_TEMPLATES = [
    ["good", "{noun}"], ["really", "good", "{noun}"], ["very", "fun", "{noun}"],
    ["not", "bad", "{noun}"], ["not", "boring", "{noun}"], ["quite", "nice", "{noun}"],
    ["love", "this", "{noun}"], ["enjoyable", "{noun}"],
]
_NEG_TEMPLATES = [
    ["bad", "{noun}"], ["really", "bad", "{noun}"], ["very", "boring", "{noun}"],
    ["not", "good", "{noun}"], ["not", "fun", "{noun}"], ["quite", "weak", "{noun}"],
    ["hate", "this", "{noun}"], ["awful", "{noun}"],
]
_PREFIXES = [[], ["overall"], ["today"], ["honestly"], ["for", "me"]]
_SUFFIXES = [[], ["overall"], ["for", "me"], ["today"]]


def _render(template: list[str], noun: str) -> list[str]:
    return [t.format(noun=noun) for t in template]


def make_dataset(n_per_label: int = 700, seed: int = 42) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for label, templates in [(1, _POS_TEMPLATES), (0, _NEG_TEMPLATES)]:
        for _ in range(n_per_label):
            noun = rng.choice(_NOUNS)
            tokens = rng.choice(_PREFIXES) + _render(rng.choice(templates), noun) + rng.choice(_SUFFIXES)
            texts.append(" ".join(tokens))
            labels.append(label)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_sklearn_baseline(name: str, ngram_range: tuple, max_iter: int, splits: dict, seed: int) -> dict:
    vec = CountVectorizer(ngram_range=tuple(ngram_range))
    x_train = vec.fit_transform(splits["train_texts"])
    x_val   = vec.transform(splits["val_texts"])
    x_test  = vec.transform(splits["test_texts"])

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(x_train, splits["y_train"])

    val_preds  = clf.predict(x_val)
    test_preds = clf.predict(x_test)

    return {
        "model": name,
        "family": "sklearn",
        "val_acc": round(accuracy_score(splits["y_val"], val_preds), 6),
        "test_acc": round(accuracy_score(splits["y_test"], test_preds), 6),
        "confusion_matrix": confusion_matrix(splits["y_test"], test_preds).tolist(),
        "classification_report": classification_report(
            splits["y_test"], test_preds,
            target_names=["negative", "positive"], output_dict=True, zero_division=0,
        ),
    }


def train_lstm(
    exp_cfg: dict,
    splits: dict,
    loaders: dict,
    seed: int,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[nn.Module, list]:
    set_seed(seed)
    amp_enabled = bool(use_amp and device.type == "cuda")
    vocab_size = splits["vocab_size"]
    model = TextLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=exp_cfg["embed_dim"],
        hidden_dim=exp_cfg["hidden_dim"],
    ).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, exp_cfg["epochs"] + 1):
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
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6), "train_acc": round(train_acc, 6),
            "val_loss":   round(val_loss,   6), "val_acc":   round(val_acc,   6),
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_dir  = Path(__file__).resolve().parent
    artifacts_dir = ensure_dir(project_dir / "artifacts")
    config = json.loads((project_dir / "config.json").read_text(encoding="utf-8"))

    device = resolve_device(config.get("device", "auto"))
    use_amp = config.get("use_amp", False)
    set_seed(config["seed"])
    ds_cfg = config["dataset"]
    texts, labels = make_dataset(n_per_label=ds_cfg["n_per_label"], seed=config["seed"])

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        texts, labels, test_size=ds_cfg["test_size"], random_state=config["seed"], stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=ds_cfg["val_size"], random_state=config["seed"], stratify=y_train_full,
    )
    splits = {
        "train_texts": x_train, "val_texts": x_val, "test_texts": x_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

    # Build vocab from training data
    text_cfg = config["text"]
    vocab, stoi = build_vocab(x_train, min_freq=text_cfg["min_freq"])
    splits["vocab_size"] = len(vocab)
    max_len = text_cfg["max_len"]

    results, histories = [], {}

    # sklearn baselines
    for bl_cfg in config["baselines"]:
        result = train_sklearn_baseline(
            bl_cfg["name"],
            bl_cfg["ngram_range"],
            bl_cfg["max_iter"],
            splits,
            seed=config["seed"],
        )
        results.append(result)
        print(f"{result['model']}: val={result['val_acc']:.4f}  test={result['test_acc']:.4f}")

    # LSTM experiments
    for exp_cfg in config["experiments"]:
        loaders = make_text_loaders(
            x_train,
            list(y_train),
            x_val,
            list(y_val),
            x_test,
            list(y_test),
            stoi,
            max_len,
            train_batch_size=config["train_batch_size"],
            eval_batch_size=config["eval_batch_size"],
            seed=config["seed"],
        )
        model, history = train_lstm(
            exp_cfg,
            splits,
            loaders,
            seed=config["seed"],
            device=device,
            use_amp=use_amp,
        )
        val_preds, val_targets = collect_predictions(model, loaders["val"], device=device)
        test_preds, test_targets = collect_predictions(model, loaders["test"], device=device)
        histories[exp_cfg["name"]] = history
        result = {
            "model":   exp_cfg["name"],
            "family":  "torch",
            "config":  exp_cfg,
            "val_acc":  round(accuracy_score(val_targets.numpy(), val_preds.numpy()), 6),
            "test_acc": round(accuracy_score(test_targets.numpy(), test_preds.numpy()), 6),
            "confusion_matrix": confusion_matrix(test_targets.numpy(), test_preds.numpy()).tolist(),
            "classification_report": classification_report(
                test_targets.numpy(), test_preds.numpy(),
                target_names=["negative", "positive"], output_dict=True, zero_division=0,
            ),
        }
        results.append(result)
        print(f"{result['model']}: val={result['val_acc']:.4f}  test={result['test_acc']:.4f}")

    sorted_results = sorted(results, key=lambda r: (r["test_acc"], r["val_acc"]), reverse=True)

    config_used = {
        **config,
        "runtime": {
            "device": str(device),
            "amp_enabled": bool(use_amp and device.type == "cuda"),
        },
    }
    save_json(config_used,    artifacts_dir / "config_used.json")
    save_json(histories,      artifacts_dir / "histories.json")
    save_json(sorted_results, artifacts_dir / "results.json")

    print("\nSaved artifacts to", artifacts_dir)
    print("\nLeaderboard:")
    for row in sorted_results:
        print(f"  {row['model']}: val={row['val_acc']:.4f}  test={row['test_acc']:.4f}")


if __name__ == "__main__":
    main()
