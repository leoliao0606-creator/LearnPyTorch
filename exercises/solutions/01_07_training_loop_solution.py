"""Reference solution for 01_07_training_loop.py."""

import torch
import torch.nn as nn


model = nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
x = torch.randn(8, 4)
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

logits = model(x)
loss = loss_fn(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("exercise 1 loss =", loss.item())

print("exercise 2:")
print("- model.train() enables training behavior such as Dropout randomness.")
print("- model.eval() switches layers such as Dropout and BatchNorm to inference behavior.")


def batch_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


print("exercise 3 batch_accuracy =", batch_accuracy(logits, y))
