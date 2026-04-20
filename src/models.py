"""Reusable model definitions for notebooks and projects."""

from __future__ import annotations

import torch.nn as nn


class TabularMLP(nn.Module):
    """A compact MLP for tabular classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.0, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TextLSTMClassifier(nn.Module):
    """A simple LSTM classifier for tokenized text."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 48, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        return self.fc(h_n[-1])
