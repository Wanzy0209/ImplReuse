import torch
import torch._inductor.config as config
from torch.library import impl_abstract, define, impl

# Configure Inductor to use cpp_wrapper, which is the context of the bug
config.cpp_wrapper = True

# Define a custom operator that takes a non-tensor argument (Scalar)
# The bug specifically mentions redundant H2D-D2H memcpy for non-tensor arguments
define("test_ns::custom_op", "(Tensor x, Scalar scalar) -> Tensor")

# Use the similar API: torch.library.impl_abstract
# This registers the FakeTensor implementation required for torch.compile
@impl_abstract("test_ns::custom_op")
def custom_op_abstract(x, scalar):
    # For the abstract impl, we just return a tensor with the same shape as x
    return x

# Register a concrete implementation for CUDA
@impl("test_ns::custom_op", "CUDA")
def custom_op_cuda(x, scalar):
    return x + scalar

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Compile the forward pass
        self.forward = torch.compile(self.forward)

    def forward(self, x):
        # Call the custom op with a scalar argument
        # This triggers the code path in Inductor where non-tensor args are wrapped
        return torch.ops.test_ns.custom_op(x, 1.5)

if torch.cuda.is_available():
    model = TestModule().cuda()

    # The bug occurs when running under a DeviceContext
    with torch.device("cuda"):
        x = torch.randn(4, 4).cuda()
        y = model(x)

        # Verify correctness
        expected = x + 1.5
        assert torch.allclose(y, expected), "Output mismatch"
else:
    print("CUDA not available, skipping test.")