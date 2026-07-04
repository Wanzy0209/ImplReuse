# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

t = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")
torch.linalg.lu_factor(t)