import torch

# Enable the configuration that triggered the bug in the original report
torch._inductor.config.combo_kernels = True

# Define the function using torch.any (the similar API) alongside other operations
# to encourage combo kernel generation.
@torch.compile
def fn(x, y, z):
    return x.sum(1), y.mean(1), z.any(1)

# Setup inputs
inps = (
    torch.rand(16, 128, device="cuda"),
    torch.rand(32, 128, device="cuda"),
    torch.rand(32, 256, device="cuda"),
)

# Execute the compiled function
compiled_res = fn(*inps)

# Verify results against eager execution to ensure correctness
eager_res = (inps[0].sum(1), inps[1].mean(1), inps[2].any(1))

assert torch.allclose(compiled_res[0], eager_res[0]), "Sum mismatch"
assert torch.allclose(compiled_res[1], eager_res[1]), "Mean mismatch"
assert torch.equal(compiled_res[2], eager_res[2]), "Any mismatch"

print("Test passed.")