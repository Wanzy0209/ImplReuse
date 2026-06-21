import torch
import sys

# Fix: Explicitly import torch._inductor to ensure it is accessible
# This resolves the AttributeError if the module exists but isn't loaded yet.
try:
    import torch._inductor
except ImportError:
    print("Skipping test: torch._inductor is not available in this environment.")
    sys.exit(0)

# Fix: Check for CUDA availability since the test uses device="cuda"
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available.")
    sys.exit(0)

# Enable the configuration that triggers the original bug
torch._inductor.config.combo_kernels = True

# Define a custom operator to leverage torch.library.get_ctx
# This API is used to access the FakeImplCtx for defining metadata
def custom_op_meta(x):
    # Leverage the similar API: torch.library.get_ctx
    ctx = torch.library.get_ctx()
    return ctx.new_zeros(x.shape, dtype=x.dtype)

def custom_op_impl(x):
    return x * 2.0

# Register the operator
torch.library.define("test_ns::custom_op(Tensor x) -> Tensor")
torch.library.register_fake("test_ns::custom_op", custom_op_meta)
torch.library.register_impl("test_ns::custom_op", custom_op_impl)

# Function to compile, combining the custom op with the bug trigger (cumsum)
@torch.compile
def fn(x, y, z):
    # Call the custom op (defined using the similar API)
    a = torch.ops.test_ns.custom_op(x)
    # Call cumsum, which requires helper functions that caused the NameError
    b = z.cumsum(1)
    return a, b

# Inputs
inps = (
    torch.rand(16, 128, device="cuda"),
    torch.rand(32, 128, device="cuda"),
    torch.rand(32, 256, device="cuda"),
)

# Run the test
# If the bug is present, this will raise NameError: _triton_helper_fn_add0 is not defined
result = fn(*inps)

# Basic assertions to verify execution
assert result[0].shape == inps[0].shape
assert result[1].shape == inps[2].shape