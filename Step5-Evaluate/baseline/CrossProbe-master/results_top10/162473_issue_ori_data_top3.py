# MOCK_TRANSLATION to tensorflow
# original source preserved below
import torch

empty_tensor = torch.tensor([], device='cuda:0')
result = torch.ops.aten.nonzero_static(empty_tensor, size=1)