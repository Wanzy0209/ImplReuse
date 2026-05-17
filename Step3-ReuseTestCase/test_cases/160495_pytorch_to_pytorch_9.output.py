import torch
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def eager_fn(x):
    return torch.sigmoid(x)

@torch.compile()
def fn(x):
    return torch.sigmoid(x)

# Test with a 0-dim complex tensor, which was the source of the crash in the original bug
x = torch.empty((), dtype=torch.complex64)

eager_fn(x)
print("eager success")
fn(x)
print("compiler success")