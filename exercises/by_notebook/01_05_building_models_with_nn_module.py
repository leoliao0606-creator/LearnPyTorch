"""01_05_building_models_with_nn_module.ipynb 对应的独立练习。

规则：
1. 不要先打开参考答案。
2. 先自己完成 TODO。
3. 做完后再和答案对照。
"""

import torch
import torch.nn as nn


# 练习 1
# 定义一个 TwoLayerMLP 类，要求：
#   - 在 __init__ 中接收 in_dim、hidden_dim、out_dim
#   - 包含两层 Linear，中间加 ReLU
#   - 正确实现 forward(x)
#
# 用 in_dim=4、hidden_dim=8、out_dim=3 实例化。
# 对 torch.randn(5, 4) 跑一次 forward，并打印输出 shape。
# 期望输出：torch.Size([5, 3])

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


# 练习 2
# 统计可训练参数量。
# 写一个 count_params(model) 函数，返回任意 nn.Module 的
# 可训练参数总数。
#
# 在上面的 TwoLayerMLP 上使用它，并手工核对：
#   第 1 层：4*8 + 8 = 40
#   第 2 层：8*3 + 3 = 27
#   总计：67

# TODO:
# def count_params(model):
#     ...
#
# print(count_params(model))  # 应该输出 67


# 练习 3
# 展示 model.train() 和 model.eval() 对 Dropout 的影响。
#
# 创建带有 Dropout(p=0.5) 的 DropoutMLP。
# 用同一个输入 torch.ones(1, 4)，
# 在 train 模式下跑 3 次，在 eval 模式下也跑 3 次。
# 观察：train 下输出应该变化，eval 下输出应该一致。

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


# 调试练习
# 下面这个模型会触发 shape 错误。找出并修复它。
# 期望行为：输入 (8, 16) -> 输出 (8, 4)

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(16, 4)   # bug is here

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# TODO: 定义 FixedModel（复制 BuggyModel 并修好 bug），然后验证：
# fixed = FixedModel()
# print(fixed(torch.randn(8, 16)).shape)  # torch.Size([8, 4])


# 总结
# 用注释回答：
# 1. super().__init__() 做了什么，为什么必须调用？
# 2. 为什么 nn.Sequential 不能直接处理 skip / residual connection？
# 3. model.parameters() 和 model.state_dict() 的区别是什么？
