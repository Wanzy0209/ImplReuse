import torch
import math
torch.set_default_device('cuda')
in_dim = 24
out_dim = 2
xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))
W = torch.normal(0, xavier_stddev, size=(in_dim, out_dim))
print(W.device)