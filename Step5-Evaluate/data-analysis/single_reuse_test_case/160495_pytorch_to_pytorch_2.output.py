import torch

# Reproduce the configuration from the original bug report
# Handle cases where torch._inductor might not be available (e.g., older PyTorch versions)
try:
    from torch._inductor import config
    config.fallback_random = True
except ImportError:
    print("Warning: torch._inductor not found. Skipping inductor config.")

torch.set_grad_enabled(False)

# Handle cases where torch._dynamo might not be available
try:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except AttributeError:
    print("Warning: torch._dynamo not found. Skipping dynamo config.")

def eager_fn(x, y):
    # Using torch.sub via the - operator
    return x - y

# Check if torch.compile is available
if hasattr(torch, 'compile'):
    @torch.compile()
    def fn(x, y):
        # Using torch.sub via the - operator
        return x - y
else:
    fn = None
    print("torch.compile is not available.")

# Setup inputs: complex64, one 1-dim tensor, one 0-dim empty tensor
x = torch.randn(1, dtype=torch.complex64)
y = torch.empty((), dtype=torch.complex64)

# Test eager mode
try:
    eager_result = eager_fn(x, y)
    print("eager success")
except Exception as e:
    print(f"eager failed: {e}")

# Test compiled mode (inductor)
if fn:
    try:
        fn(x, y)
        print("compiler success")
    except Exception as e:
        print(f"compiler failed: {e}")