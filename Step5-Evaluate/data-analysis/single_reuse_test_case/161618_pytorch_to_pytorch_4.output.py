import torch
import sys

# Check for CUDA availability since the test uses .cuda()
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    sys.exit(0)

# Handle missing torch._inductor module
try:
    import torch._inductor.config as inductor_config
except ImportError:
    print("Module 'torch._inductor' not found. This test requires PyTorch 2.0+ with inductor support. Skipping test.")
    sys.exit(0)

# Define dimensions similar to the original bug report
m = 20120
n = 512

# Create input tensor
x = torch.randn((m, n)).requires_grad_(False).cuda()

# Define function using torch.all
f = lambda x: torch.all(x)

# Compile with the same configuration that triggered the bug in the original report
with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,
):
    compiled = torch.compile(f, dynamic=False)
    result = compiled(x)

# Basic assertion to verify execution
assert isinstance(result, torch.Tensor) or isinstance(result, bool)
print(f"Test passed. Result: {result}")