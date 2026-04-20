"""Standalone exercises for 02_02_cnn_basics.ipynb.

Rules:
1. Do not open the solution file first.
2. Fill the TODO sections by yourself.
3. After finishing, compare with the solution.
"""

import torch
import torch.nn as nn


# Exercise 1 / 练习 1 — Shape tracing (no code needed)
# A grayscale image has shape (1, 28, 28).
# After each operation below, write the output shape as a comment.
#
# op1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)   # TODO: → ?
# op2 = nn.MaxPool2d(2)                               # TODO: → ?
# op3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # TODO: → ?
# op4 = nn.AdaptiveAvgPool2d(1)                       # TODO: → ?
# flatten                                             # TODO: → ?
#
# Formula: H_out = (H_in + 2*padding - kernel_size) // stride + 1
#
# After writing your answers, verify with code:

x = torch.randn(1, 1, 28, 28)
# TODO: apply each op and print shape after each step


# Exercise 2 / 练习 2 — Build a CNN block
# Build a ConvBlock class using nn.Sequential that applies:
#   Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#   BatchNorm2d(out_channels)
#   ReLU
#
# Instantiate with in_channels=1, out_channels=16.
# Verify: input (2, 1, 28, 28) → output (2, 16, 28, 28)   # same spatial size

# TODO:
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         ...
#     def forward(self, x):
#         ...
#
# block = ConvBlock(1, 16)
# print(block(torch.randn(2, 1, 28, 28)).shape)


# Exercise 3 / 练习 3 — Count Conv2d parameters
# A Conv2d(3, 64, kernel_size=3) layer — how many trainable parameters?
# Formula: C_out * (C_in * K * K + 1)   (the +1 is for bias)
#
# Write your manual calculation as a comment, then verify:

# TODO:
# manual_count = ...   # your calculation
# layer = nn.Conv2d(3, 64, kernel_size=3)
# actual_count = sum(p.numel() for p in layer.parameters())
# print(manual_count, actual_count)   # should match


# Exercise 4 / 练习 4 — End-to-end small CNN
# Build a SmallCNN for 10-class classification of (1, 28, 28) images:
#   Conv2d(1, 8, 3, padding=1) → ReLU → MaxPool2d(2)
#   Conv2d(8, 16, 3, padding=1) → ReLU → AdaptiveAvgPool2d(1)
#   flatten → Linear(16, 10)
#
# Verify: input (4, 1, 28, 28) → output (4, 10)

# TODO:
# class SmallCNN(nn.Module):
#     ...
#
# model = SmallCNN()
# print(model(torch.randn(4, 1, 28, 28)).shape)  # torch.Size([4, 10])


# Exercise 5 (debugging) / 调试练习
# The forward() below raises an error. Find the cause without running it first,
# then fix it.

class BrokenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.fc   = nn.Linear(32, 10)   # is this right?

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)

# TODO: explain what's wrong, then build FixedCNN that works on (2, 1, 10, 10)


# Reflection / 总结
# 1. Why does padding=1 with kernel_size=3 preserve spatial size?
# 2. What does AdaptiveAvgPool2d(1) do, and why is it useful?
# 3. When would you prefer MaxPool over AvgPool?
