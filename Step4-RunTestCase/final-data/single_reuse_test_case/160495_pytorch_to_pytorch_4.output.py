import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix: Handle cases where torch._inductor might not be available (e.g., older PyTorch versions)
try:
    from torch._inductor import config
    config.fallback_random = True
except (ImportError, ModuleNotFoundError):
    pass

torch.set_grad_enabled(False)

# Fix: Handle cases where torch._dynamo might not be available
try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    pass

def eager_fn(x):
    return torch.square(x)

# Fix: Ensure torch.compile is available before using it
if hasattr(torch, 'compile'):
    @torch.compile()
    def fn(x):
        return torch.square(x)
else:
    # If compile is not available, just use eager mode to avoid crash
    def fn(x):
        return torch.square(x)

# Using the scalar empty complex tensor which triggered the issue in the original binary op
x = torch.empty((), dtype=torch.complex64)

eager_fn(x)
print("eager success")
fn(x)
print("compiler success")