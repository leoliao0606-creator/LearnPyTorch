"""Utilities for whitespace-tokenized text datasets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import make_torch_generator

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize_whitespace(text: str) -> list[str]:
    """Split text on whitespace."""
    return text.split()


def build_vocab(texts: list[str], min_freq: int = 1):
    """Build a vocabulary from training texts."""
    counts: dict[str, int] = {}
    for text in texts:
        for token in tokenize_whitespace(text):
            counts[token] = counts.get(token, 0) + 1

    vocab_tokens = [token for token, count in sorted(counts.items()) if count >= min_freq]
    vocab = [PAD_TOKEN, UNK_TOKEN] + vocab_tokens
    stoi = {token: idx for idx, token in enumerate(vocab)}
    return vocab, stoi


def encode_text(text: str, stoi: dict[str, int], max_len: int) -> list[int]:
    """Encode a text into a fixed-length list of token ids."""
    unk_id = stoi[UNK_TOKEN]
    pad_id = stoi[PAD_TOKEN]
    ids = [stoi.get(token, unk_id) for token in tokenize_whitespace(text)][:max_len]
    while len(ids) < max_len:
        ids.append(pad_id)
    return ids


def build_tensor_text_dataset(texts: list[str], labels: list[int], stoi: dict[str, int], max_len: int) -> TensorDataset:
    """Convert tokenized texts into a TensorDataset."""
    features = torch.tensor([encode_text(text, stoi, max_len) for text in texts], dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(features, targets)


def make_text_loaders(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    stoi: dict[str, int],
    max_len: int,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    seed: int | None = None,
):
    """Build TensorDatasets and DataLoaders for fixed-length text classification."""
    train_ds = build_tensor_text_dataset(train_texts, train_labels, stoi, max_len)
    val_ds = build_tensor_text_dataset(val_texts, val_labels, stoi, max_len)
    test_ds = build_tensor_text_dataset(test_texts, test_labels, stoi, max_len)
    generator = make_torch_generator(seed)

    return {
        "train": DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, generator=generator),
        "val": DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False),
    }
