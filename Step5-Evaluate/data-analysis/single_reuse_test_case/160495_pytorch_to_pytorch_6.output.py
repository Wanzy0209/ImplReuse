import torch

# Fix: Handle missing torch._inductor module gracefully
try:
    from torch._inductor import config
    config.fallback_random = True
except ImportError:
    pass

torch.set_grad_enabled(False)

# Fix: Handle missing torch._dynamo attributes gracefully
try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    pass

def eager_fn(x):
    return torch.tanh(x)

@torch.compile()
def fn(x):
    return torch.tanh(x)

# Test case 1: 1-dim complex tensor
x = torch.randn(1, dtype=torch.complex64)
eager_fn(x)
print("eager success (1-dim)")
fn(x)
print("compiler success (1-dim)")

# Test case 2: 0-dim (scalar) complex tensor
# This is the specific case that triggered the error in the original bug report
y = torch.empty((), dtype=torch.complex64)
eager_fn(y)
print("eager success (0-dim)")
fn(y)
print("compiler success (0-dim)")