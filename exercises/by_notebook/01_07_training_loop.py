"""01_07_training_loop.ipynb 对应的独立练习。"""

import torch
import torch.nn as nn


# 练习 1
# 补全一个训练 step：
# 1. 跑模型
# 2. 计算 loss
# 3. 清零梯度
# 4. backward
# 5. optimizer step

model = nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
x = torch.randn(8, 4)
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

# TODO:
# logits = ...
# loss = ...
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# print("loss =", float(loss))


# 练习 2
# 解释 model.train() 和 model.eval() 的区别。

# 在这里写你的答案：
# -
# -


# 练习 3
# 写一个叫 batch_accuracy(logits, targets) 的小函数，
# 返回当前 batch 的准确率（float）。

# TODO:
# def batch_accuracy(logits, targets):
#     ...
