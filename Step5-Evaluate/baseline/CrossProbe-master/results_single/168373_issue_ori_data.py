# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import nop


def torch_compile_with_custom_backend(
    module: torch.nn.Module,
):
    opt_layer = torch.compile(
        module, backend=aot_autograd(fw_compiler=nop, bw_compiler=nop), fullgraph=True
    )

    return opt_layer


class SubMod(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_a = SubMod()
        self.mod_b = SubMod()

        self.mod_a = torch_compile_with_custom_backend(self.mod_a)
        self.mod_b = torch_compile_with_custom_backend(self.mod_b)

    def forward(self, x):
        return self.mod_a(x) + self.mod_b(x)


mod = Mod()
x = torch.randn(4)
mod(x)