# Error Log — 示例记录（三个真实高频错误）

> 这是一份预填写的示范文件，展示了三个 PyTorch 学习中最常见的错误模式。
> 按照这个格式记录你自己的报错到新文件（如 error_log_01.md）。

---

## 错误 A：Linear 层 shape 不匹配

### 基本信息
- 阶段：Phase 2 — CNN
- 对应 notebook：02_03_cnn_classification.ipynb
- 场景：把 CNN 输出接 Linear 分类头时

### 报错原文
```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x10)
```

### 最小复现
```python
import torch, torch.nn as nn

class BrokenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc   = nn.Linear(256, 10)   # ← 维度填的是拍脑袋的值

    def forward(self, x):
        x = self.conv(x)      # (B, 32, H-2, W-2)
        x = x.flatten(1)      # (B, 32*(H-2)*(W-2))
        return self.fc(x)     # 崩

BrokenCNN()(torch.randn(4, 1, 10, 10))
```

### 我当时的判断
- 以为是 Conv2d 的 padding 没设对

### 真正原因
- shape 问题：没有提前算 flatten 之后的维度
- Conv2d(1, 32, 3) 作用在 (1,10,10) 上 → 输出 (32, 8, 8) → flatten = 32*8*8 = 2048

### 最终修复
```python
# 方法 1：手动算
self.fc = nn.Linear(32 * 8 * 8, 10)

# 方法 2：用 nn.AdaptiveAvgPool2d 固定输出尺寸
self.pool = nn.AdaptiveAvgPool2d(1)   # (B, 32, 1, 1)
self.fc   = nn.Linear(32, 10)

# 方法 3（调试时临时用）：先 print shape 再填数字
def forward(self, x):
    x = self.conv(x)
    print(x.shape)   # 看完再把 Linear 维度填对
    ...
```

### 我下次应该先检查什么
1. 在 forward() 里从上到下 print 每个中间 shape
2. 或者用 `torchinfo.summary(model, input_size=(1,1,10,10))`

### 是否值得加入速查表
- 是：卷积输出 H' = (H - K + 2P) // S + 1，flatten 后 = C_out * H' * W'

---

## 错误 B：Device 不一致

### 基本信息
- 阶段：Phase 4 — Engineering
- 对应 notebook：04_04_gpu_and_efficiency.ipynb
- 场景：把模型放到 GPU 之后忘记把数据也移过去

### 报错原文
```text
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cpu!
```

### 最小复现
```python
device = torch.device("cuda")
model = nn.Linear(4, 2).to(device)   # 在 GPU
x = torch.randn(8, 4)                # 在 CPU ← 忘了 .to(device)
model(x)                             # 崩
```

### 我当时的判断
- 以为是模型没有正确移到 GPU

### 真正原因
- device 问题：model.to(device) 做了，但 x 没有做

### 最终修复
```python
for xb, yb in loader:
    xb = xb.to(device)   # ← 每个 batch 都要移
    yb = yb.to(device)
    ...
```

### 我下次应该先检查什么
1. 报错一出现先 `print(x.device, next(model.parameters()).device)`
2. 写训练循环时，`.to(device)` 和 `xb, yb` 作为一个捆绑习惯

### 是否值得加入速查表
- 是：model.to(device) 一次；每个 batch 的 xb/yb 都要 .to(device)

---

## 错误 C：Loss 立即变 NaN

### 基本信息
- 阶段：Phase 1 — PyTorch Basics
- 对应 notebook：01_07_training_loop.ipynb
- 场景：第一个 epoch 第一个 batch loss 就是 nan

### 报错原文
```text
（没有报错，但 loss = nan，准确率停在 0.5）
```

### 最小复现
```python
# 数据没有标准化，特征值在 0~10000 范围
x = torch.randn(32, 4) * 5000
y = torch.randint(0, 2, (32,))
model = nn.Linear(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # lr 也偏大

logits = model(x)
loss = nn.CrossEntropyLoss()(logits, y)
print(loss)   # nan
```

### 我当时的判断
- 以为是 CrossEntropyLoss 用法不对

### 真正原因
- 数据问题 + 训练逻辑问题：数值范围太大导致 softmax 溢出（exp(10000) = inf）

### 最终修复
```python
# 1. 标准化输入
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = torch.tensor(scaler.fit_transform(x.numpy()), dtype=torch.float32)

# 2. 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 诊断工具
print(torch.isnan(logits).any())   # 看 logits 在哪步出 nan
print(x.mean(), x.std())           # 看输入分布
```

### 我下次应该先检查什么
1. `torch.isnan(loss)` 立即触发时，往前追 logits 是否 nan
2. 检查 `x.mean()` / `x.std()` — 正常范围应在 [-3, 3] 附近
3. 检查 lr — Adam 默认 1e-3 是安全起点

### 是否值得加入速查表
- 是：loss=nan 的三个高频原因：① 输入未标准化 ② lr 太大 ③ log(0)（使用了 log 但输入含0）
