import torch

a = torch.Tensor([[1, 2, 3], [1, 2, 3]])
b = torch.Tensor([[2, 1]]).view(2, -1)
c = a + b
print(a)
print(a.size())

print(b)
print(b.size())

print(c)