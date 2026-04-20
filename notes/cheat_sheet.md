# PyTorch Cheat Sheet

## Shape Formulas

### Conv2d
```text
H_out = (H_in + 2 * padding - kernel_size) // stride + 1
# Default: padding=0, stride=1
# Example: (28, 28) -> Conv2d(1, 32, 3) -> (26, 26)
# Example: (28, 28) -> Conv2d(1, 32, 3, padding=1) -> (28, 28)
```

### LSTM
```python
# Input:  (batch, seq_len, input_size)    # batch_first=True
# Output: (batch, seq_len, hidden_size)   # output at every time step
# h_n:    (num_layers, batch, hidden_size)  # final hidden state
# Last time step: h_n[-1] has shape (batch, hidden_size)
```

### Embedding
```text
nn.Embedding(vocab_size, embed_dim)
Input: (batch, seq_len) -> Output: (batch, seq_len, embed_dim)
```

## Training Loop Skeleton

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

## Common Tensor Ops

```python
x.shape
x.view(a, b)      # reshape, requires contiguous memory
x.reshape(a, b)   # reshape, handles non-contiguous tensors
x.flatten(1)      # flatten from dim=1, keep the batch dimension
x.unsqueeze(1)    # insert a new axis at dim=1
x.squeeze(1)      # remove dim=1 if its size is 1
x.permute(0, 2, 1)
x.to(device)
x.detach()
x.numpy()         # requires CPU tensor with no gradient tracking
```

## Save and Load

```python
torch.save(model.state_dict(), "checkpoint.pt")

model = MyModel(...)
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
model.eval()
```

## Debug Order

```text
1. shape   -> print(x.shape) after every major layer
2. dtype   -> print(x.dtype)
3. device  -> print(x.device, next(model.parameters()).device)
4. loss    -> print(torch.isnan(loss), torch.isinf(loss))
5. grad    -> print(param.grad)
6. data    -> print(x.mean(), x.std())
```

## Common Error Quick Reference

| Error Message | Common Cause |
| --- | --- |
| `mat1 and mat2 shapes cannot be multiplied` | `Linear.in_features` is wrong, usually after a bad flatten dimension |
| `Expected all tensors to be on the same device` | Inputs or labels were not moved to `device` |
| `loss = nan` | Unnormalized input, learning rate too large, or unstable math such as `log(0)` |
| `Expected input batch_size to match target batch_size` | Labels have an extra dimension and need `.squeeze()` |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | An in-place update such as `x += ...` broke autograd |
| `CUDA out of memory` | Batch size is too large or eval code forgot `torch.no_grad()` |

## Parameter Count

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Linear(in, out) -> in * out + out
# Conv2d(C_in, C_out, K) -> C_out * (C_in * K * K + 1)
# LSTM(input, hidden) -> 4 * (input * hidden + hidden * hidden + hidden)
```
