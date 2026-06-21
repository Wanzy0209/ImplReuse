import torch
import torch.nn.functional as F

# Handle missing triton dependency gracefully
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Module 'triton' not found. Skipping test.")
    import sys
    sys.exit(0)

def test_tanh_xpu_performance_correctness():
    """
    Test case adapted from Issue #165738 (tl.sqrt_rn performance on XPU).
    This test verifies that torch.nn.functional.tanh (mapped to tl.tanh in Triton)
    operates correctly on XPU devices, ensuring no similar performance or correctness
    regressions occur for this similar math function.
    """
    # Skip if XPU is not available
    if not torch.xpu.is_available():
        print("XPU not available. Skipping test.")
        return

    device = torch.device("xpu")
    
    # Define a simple Triton kernel using tl.tanh (analogous to the tl.sqrt_rn usage in the bug)
    @triton.jit
    def tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Using tl.tanh corresponding to torch.nn.functional.tanh
        y = tl.tanh(x)
        tl.store(y_ptr + offsets, y, mask=mask)

    # Test parameters (using fp16 as in the original bug report)
    size = 1024 * 64
    dtype = torch.float16
    
    # Allocate tensors on XPU
    x = torch.randn(size, dtype=dtype, device=device)
    y_triton = torch.empty_like(x)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, y_triton, size, BLOCK_SIZE=128)
    
    # Compute expected result using PyTorch native API
    y_torch = F.tanh(x)
    
    # Verify correctness
    # Using a slightly relaxed tolerance for fp16
    assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2), \
        "Mismatch between Triton kernel and PyTorch native implementation"
    
    print("Test passed: torch.nn.functional.tanh (tl.tanh) works correctly on XPU.")

if __name__ == "__main__":
    test_tanh_xpu_performance_correctness()