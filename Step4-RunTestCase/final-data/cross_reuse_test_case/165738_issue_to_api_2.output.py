import torch
import sys

# Handle missing triton dependency gracefully
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Skipping test: Triton is not available.")
    sys.exit(0)

# The bug report highlights a performance regression with tl.sqrt_rn on Intel XPU.
# The similar API torch.backends.nnpack.is_available is used to check for backend support.
# We reuse this pattern to ensure the test only runs on the specific hardware (XPU) 
# where the bug was reported, mimicking the availability check logic.

def is_xpu_available():
    """
    Reusing the pattern of torch.backends.nnpack.is_available
    to check for the specific backend (XPU) mentioned in the bug.
    """
    return hasattr(torch, 'xpu') and torch.xpu.is_available()

@triton.jit
def sqrt_rn_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    # The API under test from the bug report
    result = tl.sqrt_rn(x)
    tl.store(out_ptr + offsets, result, mask=mask)

def test_tl_sqrt_rn_on_xpu():
    # Leveraging the similar API pattern: Check availability before execution
    if not is_xpu_available():
        print("Skipping test: XPU is not available.")
        return

    device = torch.device("xpu")
    # Using fp16 as indicated in the bug report's kernel signature ('*fp16')
    size = 1024
    x = torch.randn(size, dtype=torch.float16, device=device)
    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    # Run the kernel. In a real performance regression test, timing would be measured here.
    sqrt_rn_kernel[grid](x, y, size, BLOCK_SIZE=128)

    # Verify correctness to ensure the kernel runs without errors
    expected = torch.sqrt(x)
    assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2), "Output mismatch"

if __name__ == "__main__":
    test_tl_sqrt_rn_on_xpu()