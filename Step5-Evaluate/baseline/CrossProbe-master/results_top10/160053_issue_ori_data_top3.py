# MOCK_TRANSLATION to tensorflow
# original source preserved below
import torch
import torch.nn.functional as F
a = torch.empty(2,2,2,2)
F.pad(a, (1,1), mode="circular")