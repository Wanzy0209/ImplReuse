# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

def addcmul_func(x, y, z):
    return x + (y * z)


x = torch.randn(128).to("xpu")
y = torch.randn(128).to("xpu")
z = torch.randn(128).to("xpu")

out = addcmul_func(x,y,z)
print("eager mode passed")

addcmul_func_compiled = torch.compile(addcmul_func)
out = addcmul_func_compiled(x,y,z)
print("torch.compile passed")