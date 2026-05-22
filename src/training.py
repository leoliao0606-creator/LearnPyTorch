"""Reusable training helpers for classification projects."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch


def run_classification_epoch(
    model,
    loader: Iterable,
    loss_fn,
    optimizer=None,
    device: str | torch.device = "cpu",
    amp_enabled: bool = False,
    grad_scaler: torch.amp.GradScaler | None = None,
):
    """Run one training or evaluation epoch for a classification model."""
    device = torch.device(device)
    amp_enabled = bool(amp_enabled and device.type == "cuda")
    is_train = optimizer is not None
    model.to(device)
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            amp_context = torch.autocast(device_type=device.type) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(xb)
                loss = loss_fn(logits, yb)

            if is_train:
                if amp_enabled and grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
            total_items += xb.size(0)

    if total_items == 0:
        raise ValueError("loader did not yield any batches")
    return total_loss / total_items, total_correct / total_items


def collect_predictions(model, loader: Iterable, device: str | torch.device = "cpu"):
    """Collect predictions and targets from a classification loader."""
    device = torch.device(device)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(yb.cpu())

    if not all_preds:
        raise ValueError("loader did not yield any batches")
    return torch.cat(all_preds), torch.cat(all_targets)
