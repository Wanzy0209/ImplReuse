# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def forward(self, x):
        x_sparse = x.to_sparse()
        result = x_sparse * 2
        return result.to_dense()

x = torch.randn(10, 10)

model = TestModel()
print("Eager output:", model(x))
print("Compiled output:", torch.compile(model)(x))