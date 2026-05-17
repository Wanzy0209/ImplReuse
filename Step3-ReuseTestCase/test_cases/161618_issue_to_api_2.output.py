import triton
import torch
import torch._inductor.config as inductor_config

# Setup inputs suitable for broadcast_tensors
# Using shapes that require broadcasting to exercise the lowering logic
a = torch.randn((4, 1, 1)).requires_grad_(False).cuda()
b = torch.randn((1, 4, 1)).requires_grad_(False).cuda()
c = torch.randn((1, 1, 4)).requires_grad_(False).cuda()

# Function using the similar API: torch.broadcast_tensors
f = lambda a, b, c: torch.broadcast_tensors(a, b, c)

# Use the same configuration that triggered the original bug in Inductor
# to ensure the similar API works correctly under the same conditions.
with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,
):
    compiled = torch.compile(f, dynamic=False)
    result = compiled(a, b, c)

# Verify correctness against eager execution
expected = torch.broadcast_tensors(a, b, c)
for res, exp in zip(result, expected):
    assert torch.allclose(res, exp), "Compiled output does not match eager output"
    assert res.shape == exp.shape, f"Shape mismatch: {res.shape} vs {exp.shape}"

print("Test passed.")