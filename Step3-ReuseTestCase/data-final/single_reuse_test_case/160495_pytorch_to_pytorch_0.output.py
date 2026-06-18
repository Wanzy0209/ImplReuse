import torch
from torch._inductor import config

# Reproduce the specific configuration required to trigger the bug
config.fallback_random = True
torch.set_grad_enabled(False)
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Define inputs: one 1-dim complex tensor, one 0-dim (scalar) empty complex tensor
x = torch.randn(1, dtype=torch.complex64)
y = torch.empty((), dtype=torch.complex64)

# Test with eager mode using torch.add
try:
    eager_result = torch.add(x, y)
    print("eager success")
except Exception as e:
    print(f"eager failed: {e}")

# Test with compiled mode using torch.add
@torch.compile()
def fn(a, b):
    return torch.add(a, b)

try:
    compiled_result = fn(x, y)
    print("compiler success")
    # Verify results match if execution succeeds
    assert torch.equal(eager_result, compiled_result)
except Exception as e:
    print(f"compiler failed: {e}")