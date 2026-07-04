import torch
import torch.nn.functional as F

# Check for Triton availability
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton is not installed. Skipping test.")
    import sys
    sys.exit(0)

# Test case adapted from Issue 165738 (performance regression with tl.sqrt_rn on XPU)
# to verify the behavior and correctness of the similar API (softplus) on XPU.
# This test implements the softplus logic using Triton primitives on XPU.

@triton.jit
def softplus_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Logic derived from the similar API: torch.nn.functional.softplus
    # Implementation pattern: (a * 1.0).exp().log1p() / 1.0
    val = x * 1.0
    val = tl.exp(val)
    val = tl.log1p(val)
    val = val / 1.0
    
    tl.store(y_ptr + offsets, val, mask=mask)

def test_softplus_on_xpu():
    # Check for XPU availability
    if not torch.xpu.is_available():
        print("XPU is not available. Skipping test.")
        return

    device = torch.device("xpu")
    
    # Setup data
    # Using a size large enough to observe potential performance characteristics,
    # though this test primarily asserts correctness.
    size = 1024 * 1024 
    x = torch.randn(size, device=device, dtype=torch.float16)
    y_triton = torch.empty_like(x)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    # Using a block size typical for XPU workloads
    softplus_kernel[grid](x, y_triton, size, BLOCK_SIZE=1024)
    
    # Synchronize to ensure kernel completion
    torch.xpu.synchronize()
    
    # Reference implementation using PyTorch
    y_torch = F.softplus(x)
    
    # Assertion
    # Using a slightly relaxed tolerance for fp16 math operations
    assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-2), \
        f"Triton kernel output differs from PyTorch reference. Max diff: {torch.max(torch.abs(y_triton - y_torch))}"
    
    print("Test passed: Softplus kernel on XPU is correct.")

if __name__ == "__main__":
    test_softplus_on_xpu()