# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from torch import tensor
import torch

torch.aminmax(torch.tensor([1, -3, 5]))
torch.return_types.aminmax(
    min=tensor(-3),
    max=tensor(5)
)