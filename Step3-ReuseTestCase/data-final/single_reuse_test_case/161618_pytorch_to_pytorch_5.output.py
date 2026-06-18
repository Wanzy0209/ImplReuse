import torch
import torch._inductor.config as inductor_config

# Setup data similar to the original bug report's scale
# torch.any works on a single input, so we adapt the tensor creation
x = torch.randn((20120, 1536)).cuda()

# Define function using the similar API: torch.any
# Replacing the original torch.addmm call site
f = lambda x: torch.any(x)

# Compile with the same Inductor configurations as the original bug report
with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,
):
    compiled = torch.compile(f, dynamic=False)
    
    # Run the compiled function
    compiled_result = compiled(x)
    
    # Run the eager function for verification
    eager_result = f(x)
    
    # Assertion to ensure the compiled result matches the eager result
    assert compiled_result == eager_result, f"Results differ: compiled={compiled_result}, eager={eager_result}"
    
    print("Test passed.")