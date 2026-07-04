# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)
B = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)

@torch.compile
def linear(weight, input):
    return torch.mm(input, weight, out_dtype=torch.float32)

linear(A, B)