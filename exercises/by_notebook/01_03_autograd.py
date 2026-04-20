"""Standalone exercises for 01_03_autograd.ipynb.

Rules:
1. Do not open the solution file first.
2. Fill the TODO sections by yourself.
3. After finishing, compare with the solution.
"""

import torch


# Build a tensor x = [1.0, 2.0, 3.0] that tracks gradients.
# Compute y = (x ** 2).sum(), call backward(), and print x.grad.

# TODO:
# x = ...
# y = ...
# y.backward()
# print("x.grad =", ...)


# Create a tensor a with requires_grad=True.
# Compute b = a * 3 and c = b.detach().
# Print whether a, b, and c track gradients.

# TODO:
# a = ...
# b = ...
# c = ...
# print(...)


# Show gradient accumulation:
# 1. create a scalar parameter w
# 2. compute loss_1 = (w - 2) ** 2 and backward()
# 3. compute loss_2 = (w + 1) ** 2 and backward() again without zeroing the grad
# 4. print w.grad after each backward

# TODO:
# w = ...
# ...


# In your own words:
# 1. What problem does detach() solve?
# 2. Why can gradients accumulate if we do not zero them?
