"""03_02_lstm_intro.ipynb 对应的独立练习。

规则：
1. 不要先打开参考答案。
2. 先自己完成 TODO。
3. 做完后再和答案对照。
"""

import torch
import torch.nn as nn


# 练习 1 — 理解 LSTM 的输出 shape
# 创建一个 input_size=8、hidden_size=16、batch_first=True 的 LSTM。
# 输入一个 batch，其中有 4 条序列，每条长度是 10。
# 打印 output、h_n、c_n 的 shape。
# 然后在注释里回答：
#   - output[:, -1, :] 表示什么？
#   - h_n[-1] 表示什么？
#   - 它们一样吗？

# TODO:
# lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
# x = torch.randn(4, 10, 8)
# output, (h_n, c_n) = lstm(x)
# print(output.shape, h_n.shape, c_n.shape)


# 练习 2 — 取最后一层 hidden state
# 写一个 last_hidden(lstm, x) 函数，要求：
#   1. 让 x 通过一个 LSTM
#   2. 返回最后一层的 hidden state，shape 为 (batch, hidden_size)
#
# 验证：输入 (4, 10, 8) -> 输出 (4, 16)

# TODO:
# def last_hidden(lstm, x):
#     ...
#
# lstm = nn.LSTM(8, 16, batch_first=True)
# print(last_hidden(lstm, torch.randn(4, 10, 8)).shape)  # (4, 16)


# 练习 3 — LSTMClassifier
# 构建一个 LSTMClassifier 模块：
#   - __init__(vocab_size, embed_dim, hidden_size, num_classes)
#   - 包含 Embedding 层 + LSTM + Linear head
#   - forward 接收 shape 为 (batch, seq_len) 的整数 token id
#   - 返回 shape 为 (batch, num_classes) 的 logits
#
# 用 vocab_size=100、embed_dim=16、hidden_size=32、num_classes=2 验证：
#   输入 (8, 20) -> 输出 (8, 2)

# TODO:
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
#         ...
#     def forward(self, x):
#         ...
#
# model = LSTMClassifier(100, 16, 32, 2)
# token_ids = torch.randint(0, 100, (8, 20))
# print(model(token_ids).shape)  # torch.Size([8, 2])


# 练习 4 — 最小训练循环
# 用你自己的 LSTMClassifier 在随机数据上训练 20 步。
# 每 5 步打印一次 loss。loss 应该下降，至少不要爆炸。

# TODO:
# model = LSTMClassifier(100, 16, 32, 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
#
# for step in range(20):
#     xb = torch.randint(0, 100, (16, 20))
#     yb = torch.randint(0, 2, (16,))
#     logits = model(xb)
#     loss = loss_fn(logits, yb)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (step + 1) % 5 == 0:
#         print(f"step {step+1}: loss={loss.item():.4f}")


# 调试练习
# 下面这个模型虽然能运行，但不管输入是什么，都会给出几乎一样的预测。
# 为什么？你会怎么修？

class BrokenLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 32)   # batch_first defaults to False
        self.fc   = nn.Linear(32, 2)

    def forward(self, x):
        # 假设 x 的 shape 是 (batch, seq_len, embed_dim)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# TODO: 解释这个 bug。它会直接报错，还是会静默地产生错误结果？
# 提示：看一下 batch_first=False 时 output 的 shape 会变成什么。


# 总结
# 1. LSTM 里的 output 和 h_n 有什么区别？
# 2. 多层 LSTM 为什么通常取 h_n[-1] 而不是 h_n[0]？
# 3. 调用 lstm(x) 时如果不传 h_0，hidden state 会怎样？
