import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental import _attention

# Reproduce gradient diff issue
# (Simplified example - actual reproduction requires ROCm setup)
grad_key = torch.randn(10, 10)
b = grad_key.clone()
logsumexp = torch.randn(10, 10)
# Missing scaling: logsumexp /= 0.6931471805599453