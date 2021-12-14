import torch

x = torch.rand(2, 3)
y = x[0]
n = x.size(1)
print(n)
