import torch
import torch._inductor.config as inductor_config

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