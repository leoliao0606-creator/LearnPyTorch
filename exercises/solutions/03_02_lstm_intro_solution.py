"""Solutions for 03_02_lstm_intro.ipynb exercises.

Do not open before attempting the exercises yourself.
"""

import torch
import torch.nn as nn


# Exercise 1 solution
lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(4, 10, 8)
output, (h_n, c_n) = lstm(x)
print(output.shape)  # (4, 10, 16) — hidden state at every time step
print(h_n.shape)     # (1, 4, 16)  — last time step, 1 layer
print(c_n.shape)     # (1, 4, 16)  — cell state

# output[:, -1, :] → (4, 16) — last time step from output tensor
# h_n[-1]          → (4, 16) — last layer's hidden state
# They are the same for a single-layer LSTM (numerically equal).
print(torch.allclose(output[:, -1, :], h_n[-1]))  # True


# Exercise 2 solution
def last_hidden(lstm_module: nn.LSTM, x: torch.Tensor) -> torch.Tensor:
    _, (h_n, _) = lstm_module(x)
    return h_n[-1]   # (batch, hidden_size)


lstm2 = nn.LSTM(8, 16, batch_first=True)
print(last_hidden(lstm2, torch.randn(4, 10, 8)).shape)  # (4, 16)


# Exercise 3 solution
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)         # (B, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embedded)    # h_n: (1, B, hidden_size)
        return self.fc(h_n[-1])              # (B, num_classes)


model = LSTMClassifier(100, 16, 32, 2)
token_ids = torch.randint(0, 100, (8, 20))
print(model(token_ids).shape)  # torch.Size([8, 2])


# Exercise 4 solution
model4 = LSTMClassifier(100, 16, 32, 2)
optimizer = torch.optim.Adam(model4.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(20):
    xb = torch.randint(0, 100, (16, 20))
    yb = torch.randint(0, 2, (16,))
    logits = model4(xb)
    loss = loss_fn(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 5 == 0:
        print(f"step {step+1}: loss={loss.item():.4f}")


# Exercise 5 — Bug explanation
# BrokenLSTM uses batch_first=False (the default).
# When x arrives as (batch, seq_len, embed_dim), PyTorch interprets it as
# (seq_len, batch, embed_dim) — i.e. seq_len and batch are swapped.
# output shape becomes (batch, seq_len, 32) where the first dim is treated as
# seq_len, so output[:, -1, :] indexes along the wrong dimension.
# The model doesn't crash, but it reads from the wrong position and produces
# nonsensical results — a silent shape bug, the worst kind.
#
# Fix: add batch_first=True.
class FixedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 32, batch_first=True)   # fixed
        self.fc   = nn.Linear(32, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


# Reflection answers
# 1. output contains hidden states for every time step: (B, T, H).
#    h_n contains only the last time step's hidden state: (layers, B, H).
#    Use output when you need attention over all steps; use h_n when you just
#    want a sequence summary.
#
# 2. For a multi-layer LSTM, h_n has shape (num_layers, B, H).
#    h_n[-1] selects the top (last) layer's hidden state, which has seen
#    all layers of processing. h_n[0] would be the first layer.
#
# 3. When h_0 is not passed, PyTorch initializes it to zeros. This is the
#    standard behavior for sequence classification (each sample starts fresh).
