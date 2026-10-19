# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def eager_fn(x, y):
    return x + y

@torch.compile()
def fn(x, y):
    return x + y

x = torch.randn(1, dtype=torch.complex64)
y = torch.empty((), dtype=torch.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")