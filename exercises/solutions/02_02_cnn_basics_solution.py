"""Solutions for 02_02_cnn_basics.ipynb exercises.

Do not open before attempting the exercises yourself.
"""

import torch
import torch.nn as nn


# Exercise 1 solution — Shape tracing
# Input: (1, 28, 28)
# Conv2d(1, 16, 3, padding=0): H = (28+0-3)//1+1 = 26  → (16, 26, 26)
# MaxPool2d(2):                 H = 26//2 = 13          → (16, 13, 13)
# Conv2d(16, 32, 3, padding=1): H = (13+2-3)//1+1 = 13 → (32, 13, 13)
# AdaptiveAvgPool2d(1):                                  → (32, 1, 1)
# flatten(1):                                            → (32,)

x = torch.randn(1, 1, 28, 28)
op1 = nn.Conv2d(1, 16, 3, padding=0)
op2 = nn.MaxPool2d(2)
op3 = nn.Conv2d(16, 32, 3, padding=1)
op4 = nn.AdaptiveAvgPool2d(1)

out = op1(x); print("after conv1:", out.shape)   # (1, 16, 26, 26)
out = op2(out); print("after pool1:", out.shape) # (1, 16, 13, 13)
out = op3(out); print("after conv2:", out.shape) # (1, 32, 13, 13)
out = op4(out); print("after pool2:", out.shape) # (1, 32, 1, 1)
out = out.flatten(1); print("after flatten:", out.shape) # (1, 32)


# Exercise 2 solution — ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


block = ConvBlock(1, 16)
print(block(torch.randn(2, 1, 28, 28)).shape)  # (2, 16, 28, 28)


# Exercise 3 solution — Count Conv2d parameters
# Conv2d(3, 64, 3): weights = 64 * (3 * 3 * 3) = 1728, bias = 64 → total = 1792
manual_count = 64 * (3 * 3 * 3 + 1)
layer = nn.Conv2d(3, 64, kernel_size=3)
actual_count = sum(p.numel() for p in layer.parameters())
print(manual_count, actual_count)  # 1792 1792


# Exercise 4 solution — SmallCNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


model = SmallCNN()
print(model(torch.randn(4, 1, 28, 28)).shape)  # torch.Size([4, 10])


# Exercise 5 fix
# Bug: after Conv2d(1, 32, 3) on (2,1,10,10), output is (2,32,8,8).
# flatten(1) → (2, 32*8*8) = (2, 2048). Linear(32, 10) expects 32 features → mismatch.
# Fix: use AdaptiveAvgPool2d(1) before flatten, then Linear(32, 10).

class FixedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)         # (B, 32, 1, 1)
        x = x.flatten(1)         # (B, 32)
        return self.fc(x)


fixed = FixedCNN()
print(fixed(torch.randn(2, 1, 10, 10)).shape)  # torch.Size([2, 10])


# Reflection answers
# 1. With kernel_size=3 and padding=1: H_out = (H+2-3)//1+1 = H. The +2 padding
#    exactly compensates for the -2 reduction from a kernel_size=3 convolution.
#
# 2. AdaptiveAvgPool2d(1) averages all spatial positions into a single value
#    per channel, giving output (B, C, 1, 1) regardless of input spatial size.
#    Useful because the Linear head can be fixed even with variable input sizes.
#
# 3. MaxPool preserves the strongest activations (sharp edges, textures).
#    AvgPool computes a spatial mean (smoother). MaxPool is more common in
#    early CNN layers; AvgPool (or AdaptiveAvgPool) is common as a final pooling
#    before the classifier head.
