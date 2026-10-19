# MOCK_TRANSLATION to tensorflow
# original source preserved below
import torch
import torch.nn as nn

class JITCrash(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

model = JITCrash()
scripted = torch.jit.script(model)