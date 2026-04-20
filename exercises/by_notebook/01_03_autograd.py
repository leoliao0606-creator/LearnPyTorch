"""01_03_autograd.ipynb 对应的独立练习。

规则：
1. 不要先打开参考答案。
2. 先自己完成 TODO。
3. 做完后再和答案对照。
"""

import torch


# 练习 1
# 构造一个会追踪梯度的张量 x = [1.0, 2.0, 3.0]。
# 计算 y = (x ** 2).sum()，调用 backward()，并打印 x.grad。

# TODO:
# x = ...
# y = ...
# y.backward()
# print("x.grad =", ...)


# 练习 2
# 创建一个 requires_grad=True 的张量 a。
# 计算 b = a * 3 和 c = b.detach()。
# 打印 a、b、c 是否会追踪梯度。

# TODO:
# a = ...
# b = ...
# c = ...
# print(...)


# 练习 3
# 展示梯度累积：
# 1. 创建一个标量参数 w
# 2. 计算 loss_1 = (w - 2) ** 2 并 backward()
# 3. 不清零梯度，再计算 loss_2 = (w + 1) ** 2 并 backward()
# 4. 每次 backward 后都打印 w.grad

# TODO:
# w = ...
# ...


# 总结
# 用你自己的话回答：
# 1. detach() 解决了什么问题？
# 2. 为什么如果不清零梯度，梯度会累积？
