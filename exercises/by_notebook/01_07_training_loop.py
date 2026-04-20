"""Standalone exercises for 01_07_training_loop.ipynb."""

import torch
import torch.nn as nn


# Exercise 1 / 练习 1
# Complete one training step:
# 1. run the model
# 2. compute the loss
# 3. zero gradients
# 4. backward
# 5. optimizer step

model = nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
x = torch.randn(8, 4)
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

# TODO:
# logits = ...
# loss = ...
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# print("loss =", float(loss))


# Exercise 2 / 练习 2
# Explain the difference between model.train() and model.eval().

# Write your answer here:
# -
# -


# Exercise 3 / 练习 3
# Write a small function called batch_accuracy(logits, targets)
# that returns the batch accuracy as a float.

# TODO:
# def batch_accuracy(logits, targets):
#     ...
