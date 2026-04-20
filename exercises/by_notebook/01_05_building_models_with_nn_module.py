"""Standalone exercises for 01_05_building_models_with_nn_module.ipynb.

Rules:
1. Do not open the solution file first.
2. Fill the TODO sections by yourself.
3. After finishing, compare with the solution.
"""

import torch
import torch.nn as nn


# Define a class TwoLayerMLP that:
#   - accepts in_dim, hidden_dim, out_dim in __init__
#   - has two Linear layers with ReLU in between
#   - implements forward(x) correctly
#
# Instantiate it with in_dim=4, hidden_dim=8, out_dim=3.
# Run a forward pass on torch.randn(5, 4) and print the output shape.
# Expected: torch.Size([5, 3])

# TODO:
# class TwoLayerMLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         ...
#     def forward(self, x):
#         ...
#
# model = TwoLayerMLP(4, 8, 3)
# out = model(torch.randn(5, 4))
# print(out.shape)


# Count trainable parameters.
# Write a function count_params(model) that returns the total number
# of trainable parameters in any nn.Module.
#
# Use it on the TwoLayerMLP above and verify manually:
#   Layer 1: 4*8 + 8 = 40
#   Layer 2: 8*3 + 3 = 27
#   Total: 67

# TODO:
# def count_params(model):
#     ...
#
# print(count_params(model))  # should print 67


# Demonstrate that model.train() vs model.eval() affects Dropout.
#
# Create DropoutMLP with Dropout(p=0.5).
# Run the same input (torch.ones(1, 4)) through the model 3 times
# in train mode and 3 times in eval mode.
# Observe: outputs should vary in train, be identical in eval.

# TODO:
# class DropoutMLP(nn.Module):
#     ...
#
# model_d = DropoutMLP(in_dim=4, hidden_dim=8, out_dim=2)
# x = torch.ones(1, 4)
# model_d.train()
# print([model_d(x).tolist() for _ in range(3)])
# model_d.eval()
# print([model_d(x).tolist() for _ in range(3)])


# The model below raises a shape error. Find and fix it.
# Expected behavior: input (8, 16) → output (8, 4)

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(16, 4)   # bug is here

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# TODO: define FixedModel (copy BuggyModel and fix the bug), then verify:
# fixed = FixedModel()
# print(fixed(torch.randn(8, 16)).shape)  # torch.Size([8, 4])


# Answer in comments:
# 1. What does super().__init__() do and why is it required?
# 2. Why can't nn.Sequential handle a skip/residual connection?
# 3. What is the difference between model.parameters() and model.state_dict()?
