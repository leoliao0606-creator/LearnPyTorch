"""Utilities for whitespace-tokenized text datasets."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

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
