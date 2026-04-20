# Error Log Example

This file shows three common PyTorch failure patterns and the kind of record you should keep for your own debugging history.

## Error A: Linear Layer Shape Mismatch

### Basics

- Phase: Phase 2 - CNN
- Notebook: `02_03_cnn_classification.ipynb`
- Context: connecting a CNN output to a `Linear` classifier head

### Original Error

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x10)
```

### Minimal Reproduction

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

### First Guess

- I thought the problem was incorrect padding in `Conv2d`.

### Root Cause

- The flattened feature dimension was never computed correctly.
- `Conv2d(1, 32, 3)` on `(1, 10, 10)` produces `(32, 8, 8)`.
- After flattening, the real feature size is `32 * 8 * 8 = 2048`, not `256`.

### Final Fix

```python
self.fc = nn.Linear(32 * 8 * 8, 10)
```

### What I Should Check First Next Time

1. Print the tensor shape right before every `Linear` layer.
2. Confirm the flatten dimension with real data, not intuition.

## Error B: Device Mismatch

### Basics

- Phase: Phase 1 or later
- Notebook: any training loop notebook
- Context: model on GPU, batch still on CPU

### Original Error

```text
RuntimeError: Expected all tensors to be on the same device
```

### Minimal Reproduction

```python
model = model.to("cuda")
xb, yb = next(iter(train_loader))
logits = model(xb)
```

### First Guess

- I thought CUDA itself was misconfigured.

### Root Cause

- The model was moved to GPU, but `xb` and `yb` were not.

### Final Fix

```python
xb, yb = xb.to(device), yb.to(device)
logits = model(xb)
```

### What I Should Check First Next Time

1. Print `xb.device`.
2. Print `next(model.parameters()).device`.

## Error C: `loss = nan`

### Basics

- Phase: Phase 4 or capstone work
- Notebook: any experiment notebook
- Context: unstable training after changing preprocessing or learning rate

### Original Error

```text
loss = nan
```

### First Guess

- I assumed the optimizer was broken.

### Root Cause

- The input scale changed, the learning rate was too aggressive, or unstable operations created invalid values.

### Final Fix

```python
print(x.mean(), x.std())
print(torch.isnan(loss), torch.isinf(loss))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### What I Should Check First Next Time

1. Whether the input distribution is reasonable.
2. Whether the loss becomes `nan` on the first step or only later.
3. Whether the learning rate is too large for the current setup.
