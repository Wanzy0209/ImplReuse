import torch
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import nop

# Leveraging the pattern from the similar API (tf.autograph.trace)
# to log information during the compilation/tracing phase.
def trace(*args):
    """Traces argument information at compilation time."""
    print(*args)

def torch_compile_with_custom_backend(module: torch.nn.Module):
    # The bug: a fresh backend function is created on every call.
    # We use trace to log the object ID to demonstrate the mismatch.
    backend = aot_autograd(fw_compiler=nop, bw_compiler=nop)
    trace(f"Backend created with ID: {id(backend)}")
    
    opt_layer = torch.compile(
        module, backend=backend, fullgraph=True
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

        # Calling the helper twice creates two different backend objects
        self.mod_a = torch_compile_with_custom_backend(self.mod_a)
        self.mod_b = torch_compile_with_custom_backend(self.mod_b)

    def forward(self, x):
        return self.mod_a(x) + self.mod_b(x)

if __name__ == "__main__":
    mod = Mod()
    x = torch.randn(4)
    mod(x)
    
    # Expected behavior (Bug): The trace output will show two different IDs.
    # This confirms that torch.compile sees two different backends and triggers recompilation.