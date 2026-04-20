"""Reference solution for 01_03_autograd.py."""

import torch


x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print("exercise 1 x.grad =", x.grad)

a = torch.tensor([2.0, -1.0], requires_grad=True)
b = a * 3
c = b.detach()
print("exercise 2 requires_grad:", a.requires_grad, b.requires_grad, c.requires_grad)

w = torch.tensor(1.0, requires_grad=True)
loss_1 = (w - 2) ** 2
loss_1.backward()
print("exercise 3 grad after loss_1 =", w.grad.item())
loss_2 = (w + 1) ** 2
loss_2.backward()
print("exercise 3 grad after loss_2 =", w.grad.item())
