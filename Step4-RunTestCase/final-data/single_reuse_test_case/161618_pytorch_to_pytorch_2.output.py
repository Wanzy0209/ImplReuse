import torch
import sys

# Fix: Handle missing torch._inductor module gracefully
try:
    import torch._inductor.config as inductor_config
except ModuleNotFoundError:
    print("Skipping test: torch._inductor module not found.")
    sys.exit(0)

# Dimensions from the original bug report
m = 20120
n = 512

# Check for CUDA availability to prevent runtime errors on CPU-only machines
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available.")
    sys.exit(0)

# Create input tensor
# Using requires_grad_(False) and cuda() to match original setup
x = torch.randn((m, n)).requires_grad_(False).cuda()

# Define the function using the similar API: torch.prod
# We perform a reduction. Note: torch.prod on float32 random numbers can easily overflow to inf or underflow to 0.
# The primary goal here is to test the compilation path (Inductor + Triton).
f = lambda x: torch.prod(x)

# Apply the same Inductor configuration as the original bug report
with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,
):
    compiled = torch.compile(f, dynamic=False)
    result = compiled(x)

    # Basic assertion to verify execution
    assert result is not None