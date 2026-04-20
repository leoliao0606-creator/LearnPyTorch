"""Utilities for small tabular classification datasets."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_breast_cancer_splits(seed: int = 42, test_size: float = 0.2, val_size: float = 0.2) -> dict[str, Any]:
    """Load and standardize the breast cancer dataset with train/val/test splits."""
    data = load_breast_cancer(as_frame=True)
    frame = data.frame.copy()
    frame.rename(columns={"target": "label"}, inplace=True)

    x = frame.drop(columns=["label"])
    y = frame["label"]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return {
        "feature_names": list(x.columns),
        "target_names": [str(name) for name in data.target_names],
        "x_train": x_train_scaled,
        "x_val": x_val_scaled,
        "x_test": x_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
    }


def make_tabular_loaders(split_data: dict[str, Any], train_batch_size: int = 32, eval_batch_size: int = 64):
    """Convert split arrays into PyTorch DataLoaders."""
    train_ds = TensorDataset(
        torch.tensor(split_data["x_train"], dtype=torch.float32),
        torch.tensor(split_data["y_train"], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(split_data["x_val"], dtype=torch.float32),
        torch.tensor(split_data["y_val"], dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(split_data["x_test"], dtype=torch.float32),
        torch.tensor(split_data["y_test"], dtype=torch.long),
    )

    return {
        "train": DataLoader(train_ds, batch_size=train_batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False),
    }
