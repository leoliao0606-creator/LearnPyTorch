# PyTorch 速查表

## 形状公式

### Conv2d
```text
H_out = (H_in + 2 * padding - kernel_size) // stride + 1
# 默认: padding=0, stride=1
# 例: (28, 28) -> Conv2d(1, 32, 3) -> (26, 26)
# 例: (28, 28) -> Conv2d(1, 32, 3, padding=1) -> (28, 28)
```

### LSTM
```python
# 输入:  (batch, seq_len, input_size)    # batch_first=True
# 输出:  (batch, seq_len, hidden_size)   # 每个时间步的输出
# h_n:   (num_layers, batch, hidden_size)  # 最后一个时间步的隐状态
# 最后时间步: h_n[-1] 的 shape 是 (batch, hidden_size)
```

### Embedding
```text
nn.Embedding(vocab_size, embed_dim)
输入: (batch, seq_len) -> 输出: (batch, seq_len, embed_dim)
```

## 训练循环骨架

```python
model.train()
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    logits = model(xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        ...
```

## 常用 Tensor 操作

```python
x.shape
x.view(a, b)      # reshape，要求连续内存
x.reshape(a, b)   # reshape，可处理非连续张量
x.flatten(1)      # 从 dim=1 开始展平，保留 batch 维
x.unsqueeze(1)    # 在 dim=1 插入一个新轴
x.squeeze(1)      # 如果 dim=1 的大小为 1，则移除这一维
x.permute(0, 2, 1)
x.to(device)
x.detach()
x.numpy()         # 需要张量在 CPU 且不跟踪梯度
```

## 保存与加载

```python
torch.save(model.state_dict(), "checkpoint.pt")

model = MyModel(...)
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
model.eval()
```

## 调试顺序

```text
1. shape   -> print(x.shape)，看每一层之后的形状
2. dtype   -> print(x.dtype)
3. device  -> print(x.device, next(model.parameters()).device)
4. loss    -> print(torch.isnan(loss), torch.isinf(loss))
5. grad    -> print(param.grad)
6. data    -> print(x.mean(), x.std())
```

## 常见错误速查

| 错误信息 | 高频原因 |
| --- | --- |
| `mat1 and mat2 shapes cannot be multiplied` | `Linear.in_features` 填错，通常是 flatten 后维度算错了 |
| `Expected all tensors to be on the same device` | 输入或标签没有移动到 `device` |
| `loss = nan` | 输入未标准化、学习率过大，或出现了 `log(0)` 这类不稳定计算 |
| `Expected input batch_size to match target batch_size` | 标签多了一维，需要 `.squeeze()` |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | 像 `x += ...` 这样的 in-place 操作破坏了 autograd |
| `CUDA out of memory` | batch size 太大，或者 eval 阶段忘了 `torch.no_grad()` |

## 参数量计算

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Linear(in, out) -> in * out + out
# Conv2d(C_in, C_out, K) -> C_out * (C_in * K * K + 1)
# LSTM(input, hidden) -> 4 * (input * hidden + hidden * hidden + hidden)
```
