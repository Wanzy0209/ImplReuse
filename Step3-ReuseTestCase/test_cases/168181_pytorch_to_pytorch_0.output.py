import torch
import triton
import triton.language as tl

# Check for CUDA availability as this test requires a GPU
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    exit(0)

# Define the Triton kernel used in the bug report
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def test_triton_kernel_to_cpu():
    """
    Test case for Issue 168181: [inductor] User-defined Triton kernel output + .cpu() correctness issue.
    Verifies that torch.compile correctly handles the synchronization and data transfer
    when a user-defined Triton kernel writes to a tensor that is immediately moved to CPU.
    """
    # The function to be compiled
    def f(x, y):
        out = torch.zeros_like(x)
        # Call the user-defined Triton kernel
        # Grid (1,), n_elements=16 (4x4), BLOCK_SIZE=16
        add_kernel[(1,)](x, y, out, 16, 16)
        # The problematic operation: moving to CPU immediately after kernel launch
        out_cpu = out.cpu() + 1
        return out_cpu

    # Setup inputs on GPU
    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')

    # Eager execution
    eager_out = f(x, y)

    # Compiled execution using torch.compile
    compiled_out = torch.compile(f)(x, y)

    # Verify correctness
    assert torch.allclose(eager_out, compiled_out), "Mismatch between eager and compiled output"
    print("Test passed.")

if __name__ == "__main__":
    test_triton_kernel_to_cpu()