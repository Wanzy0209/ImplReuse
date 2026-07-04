import torch

# Reproduce the configuration from the original bug report
try:
    from torch._inductor import config
    config.fallback_random = True
except ImportError:
    pass  # Skip if torch._inductor is not available in this environment

torch.set_grad_enabled(False)

try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    pass  # Skip if torch._dynamo is not available

def eager_fn(x, y):
    # Adapted to use torch.mul (via * operator)
    return x * y

@torch.compile()
def fn(x, y):
    # Adapted to use torch.mul (via * operator)
    return x * y

# Inputs: complex tensor with shape (1,) and empty complex tensor with shape ()
x = torch.randn(1, dtype=torch.complex64)
y = torch.empty((), dtype=torch.complex64)

# Test eager execution
try:
    eager_result = eager_fn(x, y)
    print("eager success")
except Exception as e:
    print(f"eager failed: {e}")

# Test compiled execution (inductor)
try:
    compiled_result = fn(x, y)
    print("compiler success")
except Exception as e:
    print(f"compiler failed: {e}")