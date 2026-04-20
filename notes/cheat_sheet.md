# PyTorch Cheat Sheet / 速查表

## 形状公式 / Shape Formulas

### Conv2d
```
H_out = (H_in + 2*padding - kernel_size) // stride + 1
# 默认: padding=0, stride=1
# 例: (28, 28) → Conv2d(1, 32, 3) → (26, 26)
# 例: (28, 28) → Conv2d(1, 32, 3, padding=1) → (28, 28)  ← same padding
```

### LSTM
```python
# 输入:  (batch, seq_len, input_size)    # batch_first=True
# output: (batch, seq_len, hidden_size)  # 每个时间步的输出
# h_n:    (num_layers, batch, hidden_size)  # 最后时间步的隐状态
# 取最后时间步: h_n[-1]  shape: (batch, hidden_size)
```

### Embedding
```
nn.Embedding(vocab_size, embed_dim)
输入: (batch, seq_len)  → 输出: (batch, seq_len, embed_dim)
```

---

## 训练循环骨架 / Training Loop Skeleton

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

---

## 常用 Tensor 操作 / Common Tensor Ops

```python
x.shape           # torch.Size
x.view(a, b)      # reshape (要求连续内存)
x.reshape(a, b)   # reshape (自动处理非连续)
x.flatten(1)      # 从 dim=1 开始 flatten，保留 batch
x.unsqueeze(1)    # 在 dim=1 插入新轴
x.squeeze(1)      # 移除 dim=1（如果为1）
x.permute(0,2,1)  # 转置任意维度
x.to(device)      # 移动到 CPU/GPU
x.detach()        # 断开计算图
x.numpy()         # 转 numpy（须在 CPU 且无梯度）
```

---

## 保存 / 加载 / Save & Load

```python
# 保存
torch.save(model.state_dict(), "checkpoint.pt")

# 加载
model = MyModel(...)
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
model.eval()
```

---

## Debug 顺序 / Debug Order

```
1. shape  → print(x.shape) 在每层之后
2. dtype  → print(x.dtype)  # float32 vs float64 vs long
3. device → print(x.device, next(model.parameters()).device)
4. loss   → print(torch.isnan(loss), torch.isinf(loss))
5. grad   → print(param.grad) 看是否全 None 或全 0
6. data   → print(x.mean(), x.std()) 检查输入分布
```

---

## 常见错误速查 / Common Error Quick Reference

| 错误信息 | 高频原因 |
|----------|----------|
| `mat1 and mat2 shapes cannot be multiplied` | Linear in_features 填错，通常是 flatten 后维度没算对 |
| `Expected all tensors to be on the same device` | xb/yb 没有 .to(device) |
| `loss = nan` | 输入未标准化 / lr 太大 / log(0) |
| `Expected input batch_size to match target batch_size` | DataLoader 的 label 维度多了一个 1，用 .squeeze() |
| `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | 用了 x += ... 等 inplace 操作，改成 x = x + ... |
| `CUDA out of memory` | batch size 太大 / 忘记在 eval 时用 torch.no_grad() |

---

## 参数量计算 / Parameter Count

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Linear(in, out)    → in*out + out
# Conv2d(C_in, C_out, K) → C_out * (C_in * K*K + 1)
# LSTM(input, hidden)    → 4 * (input*hidden + hidden*hidden + hidden)
```
