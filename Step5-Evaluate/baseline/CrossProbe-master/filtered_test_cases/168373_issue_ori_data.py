import torch
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import nop
import functools

@functools.lru_cache
def get_backend(fw_compiler, bw_compiler):
    return aot_autograd(fw_compiler=nop, bw_compiler=nop)

def torch_compile_with_custom_backend(module):
    return torch.compile(module, backend=get_backend(nop, nop), fullgraph=True)

class SubMod(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_a = torch_compile_with_custom_backend(SubMod())
        self.mod_b = torch_compile_with_custom_backend(SubMod())
    
    def forward(self, x):
        return self.mod_a(x) + self.mod_b(x)

mod = Mod()
x = torch.randn(4)
mod(x)