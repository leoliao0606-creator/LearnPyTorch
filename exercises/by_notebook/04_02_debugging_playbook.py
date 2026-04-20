"""04_02_debugging_playbook.ipynb 对应的独立练习。

目标：训练你在运行代码之前，先阅读并找出最可能的故障点。
"""

import torch
import torch.nn as nn


# 练习 1
# 在运行之前，先找出最可能的 bug：

broken_linear = nn.Linear(5, 2)
x = torch.randn(8, 3)

# 问题：
# 为什么 broken_linear(x) 会失败？
# 你第一步应该先检查什么？


# 练习 2
# 找出 dtype 问题：

loss_fn = nn.CrossEntropyLoss()
logits = torch.randn(4, 3)
targets = torch.tensor([0.0, 1.0, 2.0, 1.0], dtype=torch.float32)

# 问题：
# 为什么 loss_fn(logits, targets) 可能会失败？
# 这里的 targets 通常应该是什么 dtype？


# 练习 3
# 写一个小工具函数，打印：
# - shape
# - dtype
# - device
#
# 函数名：
# def quick_check(name, tensor):
#     ...
