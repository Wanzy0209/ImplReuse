# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn.functional as F
a = torch.empty(2,2,2,2)
F.pad(a, (1,1), mode="circular")