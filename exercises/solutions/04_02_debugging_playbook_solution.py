"""Reference solution for 04_02_debugging_playbook.py."""

import torch
import torch.nn as nn


broken_linear = nn.Linear(5, 2)
x = torch.randn(8, 3)
print("exercise 1: broken_linear expects in_features=5 but x has last dim=3")

loss_fn = nn.CrossEntropyLoss()
logits = torch.randn(4, 3)
targets = torch.tensor([0.0, 1.0, 2.0, 1.0], dtype=torch.float32)
print("exercise 2: CrossEntropyLoss target should usually be torch.long, not float32")


def quick_check(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")


quick_check("logits", logits)
