"""Standalone exercises for 04_02_debugging_playbook.ipynb.

Goal: practice reading code and spotting likely failure points
before you run it.
"""

import torch
import torch.nn as nn


# Exercise 1 / 练习 1
# Find the most likely bug before running:

broken_linear = nn.Linear(5, 2)
x = torch.randn(8, 3)

# Question:
# Why will broken_linear(x) fail?
# What should you check first?


# Exercise 2 / 练习 2
# Find the dtype problem:

loss_fn = nn.CrossEntropyLoss()
logits = torch.randn(4, 3)
targets = torch.tensor([0.0, 1.0, 2.0, 1.0], dtype=torch.float32)

# Question:
# Why can loss_fn(logits, targets) fail?
# What dtype should targets usually have here?


# Exercise 3 / 练习 3
# Write a tiny helper that prints:
# - shape
# - dtype
# - device
#
# Function name:
# def quick_check(name, tensor):
#     ...
