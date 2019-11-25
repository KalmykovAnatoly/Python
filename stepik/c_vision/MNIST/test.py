import torch

x = torch.zeros([6000, 28, 28], dtype=torch.int32)
print(x.shape)
print(x.reshape(-1, 9).shape)
