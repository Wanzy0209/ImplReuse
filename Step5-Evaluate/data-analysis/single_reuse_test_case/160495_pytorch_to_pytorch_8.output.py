import torch

# Handle missing torch._inductor gracefully
try:
    from torch._inductor import config
    config.fallback_random = True
except ModuleNotFoundError:
    pass

torch.set_grad_enabled(False)

# Handle missing torch._dynamo config gracefully
try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    pass

def eager_fn(x):
    return torch.unique(x)

@torch.compile()
def fn(x):
    return torch.unique(x)

# Adapted test case: Using a 0-dim complex tensor as input to torch.unique
# to check if the inductor handles the view/reshape logic for complex types correctly.
x = torch.empty((), dtype=torch.complex64)

eager_fn(x)
print("eager success")
fn(x)
print("compiler success")