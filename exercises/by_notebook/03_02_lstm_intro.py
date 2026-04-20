"""Standalone exercises for 03_02_lstm_intro.ipynb.

Rules:
1. Do not open the solution file first.
2. Fill the TODO sections by yourself.
3. After finishing, compare with the solution.
"""

import torch
import torch.nn as nn


# Exercise 1 / 练习 1 — Understand LSTM output shapes
# Create an LSTM with input_size=8, hidden_size=16, batch_first=True.
# Pass a batch of 4 sequences, each 10 tokens long.
# Print the shapes of: output, h_n, c_n.
# Then answer in comments:
#   - What does output[:, -1, :] give you?
#   - What does h_n[-1] give you?
#   - Are they the same?

# TODO:
# lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
# x = torch.randn(4, 10, 8)
# output, (h_n, c_n) = lstm(x)
# print(output.shape, h_n.shape, c_n.shape)


# Exercise 2 / 练习 2 — Extract the last hidden state
# Write a function last_hidden(lstm, x) that:
#   1. runs x through an LSTM
#   2. returns the last-layer hidden state of shape (batch, hidden_size)
#
# Verify: input (4, 10, 8) → output (4, 16)

# TODO:
# def last_hidden(lstm, x):
#     ...
#
# lstm = nn.LSTM(8, 16, batch_first=True)
# print(last_hidden(lstm, torch.randn(4, 10, 8)).shape)  # (4, 16)


# Exercise 3 / 练习 3 — LSTMClassifier
# Build an LSTMClassifier module:
#   - __init__(vocab_size, embed_dim, hidden_size, num_classes)
#   - Embedding layer + LSTM + Linear head
#   - forward takes integer token ids of shape (batch, seq_len)
#   - returns logits of shape (batch, num_classes)
#
# Verify with vocab_size=100, embed_dim=16, hidden_size=32, num_classes=2:
#   input (8, 20) → output (8, 2)

# TODO:
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
#         ...
#     def forward(self, x):
#         ...
#
# model = LSTMClassifier(100, 16, 32, 2)
# token_ids = torch.randint(0, 100, (8, 20))
# print(model(token_ids).shape)  # torch.Size([8, 2])


# Exercise 4 / 练习 4 — Minimal training loop
# Using your LSTMClassifier, train it for 20 steps on random data.
# Print loss every 5 steps. Loss should decrease (or at least not blow up).

# TODO:
# model = LSTMClassifier(100, 16, 32, 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
#
# for step in range(20):
#     xb = torch.randint(0, 100, (16, 20))
#     yb = torch.randint(0, 2, (16,))
#     logits = model(xb)
#     loss = loss_fn(logits, yb)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (step + 1) % 5 == 0:
#         print(f"step {step+1}: loss={loss.item():.4f}")


# Exercise 5 (debugging) / 调试练习
# The model below runs without error but always outputs the same predictions
# regardless of input. Why? What would you fix?

class BrokenLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 32)   # batch_first defaults to False
        self.fc   = nn.Linear(32, 2)

    def forward(self, x):
        # x is assumed to be (batch, seq_len, embed_dim)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# TODO: explain the bug. Does it crash? Does it silently give wrong results?
# Hint: check what happens to output shape when batch_first=False.


# Reflection / 总结
# 1. What is the difference between output and h_n in LSTM?
# 2. Why do we take h_n[-1] instead of h_n[0] for a multi-layer LSTM?
# 3. What happens to the hidden state when we call lstm(x) without passing h_0?
