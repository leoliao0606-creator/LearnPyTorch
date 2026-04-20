"""Solutions for 01_05_building_models_with_nn_module.ipynb exercises.

Do not open before attempting the exercises yourself.
"""

import torch
import torch.nn as nn


# Exercise 1 solution
class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = TwoLayerMLP(4, 8, 3)
out = model(torch.randn(5, 4))
print(out.shape)   # torch.Size([5, 3])


# Exercise 2 solution
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Manual check: (4*8 + 8) + (8*3 + 3) = 40 + 27 = 67
print(count_params(model))  # 67


# Exercise 3 solution
class DropoutMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model_d = DropoutMLP(4, 8, 2)
x = torch.ones(1, 4)

model_d.train()
print("train mode — outputs differ across calls:")
print([model_d(x).tolist() for _ in range(3)])

model_d.eval()
print("eval mode — outputs are identical:")
print([model_d(x).tolist() for _ in range(3)])


# Exercise 4 fix
# Bug: fc2's in_features must match fc1's out_features (32, not 16)
class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)   # fixed: 32

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


fixed = FixedModel()
print(fixed(torch.randn(8, 16)).shape)  # torch.Size([8, 4])


# Reflection answers
# 1. super().__init__() calls nn.Module's __init__, which sets up internal
#    bookkeeping (parameter registry, hooks, etc.). Without it, self.fc1 = ...
#    will not be registered and model.parameters() returns nothing.
#
# 2. nn.Sequential calls modules one after another in a fixed chain.
#    A skip connection needs to add the input to an intermediate output,
#    which requires a custom forward() method.
#
# 3. model.parameters() yields live Parameter tensors — used by the optimizer.
#    model.state_dict() returns an OrderedDict of tensor values (including
#    non-trainable buffers) — used for saving and loading.
