# 错误日志示例

这个文件展示三类最常见的 PyTorch 错误记录方式。你后面记录自己真实踩过的坑时，可以直接照这个结构写。

## 错误 A: Linear 层 shape 不匹配

### 基本信息

- 阶段: Phase 2 - CNN
- 对应 notebook: `02_03_cnn_classification.ipynb`
- 场景: 把 CNN 输出接到 `Linear` 分类头时

### 原始报错

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x10)
```

### 最小复现

```python
import torch
import torch.nn as nn


class BrokenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


BrokenCNN()(torch.randn(4, 1, 10, 10))
```

### 我当时的第一判断

- 以为是 `Conv2d` 的 `padding` 配错了。

### 真正原因

- flatten 后的特征维度没有提前算对。
- `Conv2d(1, 32, 3)` 作用在 `(1, 10, 10)` 上时，会得到 `(32, 8, 8)`。
- 所以 flatten 之后的真实特征数是 `32 * 8 * 8 = 2048`，不是 `256`。

### 最终修复

```python
self.fc = nn.Linear(32 * 8 * 8, 10)
```

### 下次先检查什么

1. 在每个 `Linear` 之前先打印张量 shape。
2. 不凭感觉猜 flatten 维度，用真实数据跑一遍确认。

## 错误 B: Device 不一致

### 基本信息

- 阶段: Phase 1 及以后
- 场景: 模型在 GPU 上，batch 还留在 CPU 上

### 原始报错

```text
RuntimeError: Expected all tensors to be on the same device
```

### 最小复现

```python
model = model.to("cuda")
xb, yb = next(iter(train_loader))
logits = model(xb)
```

### 我当时的第一判断

- 以为是 CUDA 环境本身坏了。

### 真正原因

- 模型被移到了 GPU，但 `xb` 和 `yb` 没有一起移动。

### 最终修复

```python
xb, yb = xb.to(device), yb.to(device)
logits = model(xb)
```

### 下次先检查什么

1. `xb.device`
2. `next(model.parameters()).device`

## 错误 C: `loss = nan`

### 基本信息

- 阶段: Phase 4 或 capstone
- 场景: 改了预处理或学习率之后训练变得不稳定

### 原始现象

```text
loss = nan
```

### 我当时的第一判断

- 以为是优化器有问题。

### 真正原因

- 输入分布变了、学习率过大，或者出现了不稳定计算。

### 最终修复

```python
print(x.mean(), x.std())
print(torch.isnan(loss), torch.isinf(loss))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 下次先检查什么

1. 输入分布是否正常
2. `nan` 是第一步就出现，还是训练到后面才出现
3. 当前学习率是否过大
