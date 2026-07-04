import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle cases where torch._inductor is not available (e.g., older PyTorch versions or specific builds)
try:
    from torch._inductor import config
    config.fallback_random = True
except ImportError:
    pass

torch.set_grad_enabled(False)

# Handle cases where torch._dynamo config might differ or be unavailable
try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    pass

def eager_fn(x, y):
    # Replaced x + y with torch.div(x, y) to test the similar API
    return torch.div(x, y)

@torch.compile()
def fn(x, y):
    return torch.div(x, y)

x = torch.randn(1, dtype=torch.complex64)
# Adapted y to use randn instead of empty to avoid division by zero,
# while preserving the shape () and dtype complex64 which triggers the bug.
y = torch.randn((), dtype=torch.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")