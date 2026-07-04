import torch

# Reproduce the environment settings from the bug report
# Handle cases where torch._inductor is not available (e.g., older PyTorch versions)
try:
    from torch._inductor import config
    config.fallback_random = True
except ModuleNotFoundError:
    pass

torch.set_grad_enabled(False)

# Handle cases where torch._dynamo config might differ
try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    pass

def eager_fn(x, y):
    # Adapted to test torch.numel instead of torch.add
    return torch.numel(x), torch.numel(y)

@torch.compile()
def fn(x, y):
    return torch.numel(x), torch.numel(y)

# Inputs from the bug report: one complex tensor of size 1, one empty complex scalar
x = torch.randn(1, dtype=torch.complex64)
y = torch.empty((), dtype=torch.complex64)

# Execute eager mode
eager_result = eager_fn(x, y)
print("eager success:", eager_result)

# Execute compiled mode
compiled_result = fn(x, y)
print("compiler success:", compiled_result)

# Verify consistency
assert eager_result == compiled_result, f"Mismatch between eager and compiled results: {eager_result} vs {compiled_result}"