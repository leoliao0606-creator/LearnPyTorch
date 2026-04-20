"""02_02_cnn_basics.ipynb 对应的独立练习。

规则：
1. 不要先打开参考答案。
2. 先自己完成 TODO。
3. 做完后再和答案对照。
"""

import torch
import torch.nn as nn


# 练习 1 — Shape 追踪（可以先不写代码）
# 一张灰度图的 shape 是 (1, 28, 28)。
# 请在下面每一步操作后面，先写出输出 shape。
#
# op1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)   # TODO: → ?
# op2 = nn.MaxPool2d(2)                               # TODO: → ?
# op3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # TODO: → ?
# op4 = nn.AdaptiveAvgPool2d(1)                       # TODO: → ?
# flatten                                             # TODO: → ?
#
# 公式：H_out = (H_in + 2*padding - kernel_size) // stride + 1
#
# 写完之后，再用代码验证：

x = torch.randn(1, 1, 28, 28)
# TODO: 依次应用每个操作，并打印每一步的 shape


# 练习 2 — 搭一个 CNN block
# 用 nn.Sequential 构建一个 ConvBlock，包含：
#   Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#   BatchNorm2d(out_channels)
#   ReLU
#
# 用 in_channels=1、out_channels=16 实例化。
# 验证：输入 (2, 1, 28, 28) -> 输出 (2, 16, 28, 28)   # 空间尺寸保持不变

# TODO:
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         ...
#     def forward(self, x):
#         ...
#
# block = ConvBlock(1, 16)
# print(block(torch.randn(2, 1, 28, 28)).shape)


# 练习 3 — 统计 Conv2d 参数量
# 一个 Conv2d(3, 64, kernel_size=3) 层有多少可训练参数？
# 公式：C_out * (C_in * K * K + 1)   （+1 表示 bias）
#
# 先把你的手算过程写成注释，再用代码验证：

# TODO:
# manual_count = ...   # 你的手算结果
# layer = nn.Conv2d(3, 64, kernel_size=3)
# actual_count = sum(p.numel() for p in layer.parameters())
# print(manual_count, actual_count)   # 两者应该一致


# 练习 4 — 端到端小型 CNN
# 为 (1, 28, 28) 图像的 10 分类任务构建一个 SmallCNN：
#   Conv2d(1, 8, 3, padding=1) → ReLU → MaxPool2d(2)
#   Conv2d(8, 16, 3, padding=1) → ReLU → AdaptiveAvgPool2d(1)
#   flatten → Linear(16, 10)
#
# 验证：输入 (4, 1, 28, 28) -> 输出 (4, 10)

# TODO:
# class SmallCNN(nn.Module):
#     ...
#
# model = SmallCNN()
# print(model(torch.randn(4, 1, 28, 28)).shape)  # torch.Size([4, 10])


# 调试练习
# 下面这个 forward() 会报错。先不要运行，先找原因，再修复。

class BrokenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.fc   = nn.Linear(32, 10)   # is this right?

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)

# TODO: 解释哪里错了，然后构建一个能处理 (2, 1, 10, 10) 输入的 FixedCNN


# 总结
# 1. 为什么 kernel_size=3 且 padding=1 会保持空间尺寸不变？
# 2. AdaptiveAvgPool2d(1) 做了什么，为什么有用？
# 3. 什么情况下你会更偏向 MaxPool 而不是 AvgPool？
