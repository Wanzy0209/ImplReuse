import torch

# Setup data similar to the original bug report's scale
# torch.any works on a single input, so we adapt the tensor creation
x = torch.randn((20120, 1536)).cuda()

# Define function using the similar API: torch.any
# Replacing the original torch.addmm call site
f = lambda x: torch.any(x)

# Handle the missing torch._inductor module gracefully
try:
    import torch._inductor.config as inductor_config
    HAS_INDUCTOR = True
except ModuleNotFoundError:
    HAS_INDUCTOR = False
    print("Warning: torch._inductor not found. Running torch.compile without specific Inductor configurations.")

# Compile with the same Inductor configurations as the original bug report
if HAS_INDUCTOR:
    with inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        autotune_fallback_to_aten=False,
    ):
        compiled = torch.compile(f, dynamic=False)
        compiled_result = compiled(x)
else:
    # Fallback: compile without the specific patch
    compiled = torch.compile(f, dynamic=False)
    compiled_result = compiled(x)

# Run the eager function for verification
eager_result = f(x)

# Assertion to ensure the compiled result matches the eager result
assert compiled_result == eager_result, f"Results differ: compiled={compiled_result}, eager={eager_result}"

print("Test passed.")