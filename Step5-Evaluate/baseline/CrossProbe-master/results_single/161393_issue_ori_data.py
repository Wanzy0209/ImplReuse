# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def f(x):
    nz = x.nonzero()
    return nz[:-1]

out = torch.compile(f, fullgraph=True)(torch.randn(3, 4))
print(out)