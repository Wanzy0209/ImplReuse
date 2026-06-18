import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# Define a simple Triton kernel that mimics a pooling operation (identity/max)
# to test memory access safety with non-contiguous tensors.
@triton.jit
def simple_max_pool_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data from input
    # Note: We treat the tensor as a 1D buffer of physical memory.
    # This tests if the kernel can safely access the memory layout 
    # (channels_last) without illegal access.
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Perform a dummy max operation (identity)
    y = tl.max(x, x)
    
    # Store data to output
    tl.store(output_ptr + offsets, y, mask=mask)

# Define the custom operation using torch.library infrastructure
@triton_op("custom::max_pool_test", schema="(Tensor x) -> Tensor")
def custom_max_pool(x):
    # Create an output tensor with the same memory format as the input
    # to ensure the physical layout (NHWC) is preserved for the kernel.
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Use wrap_triton to execute the kernel
    # This allows the custom kernel to be traced and used within the PyTorch dispatcher
    wrap_triton(simple_max_pool_kernel).run(
        x, output, n_elements, BLOCK_SIZE=1024, grid=grid
    )
    
    return output

def test_wrap_triton_channels_last():
    """
    Test case to verify that a custom kernel wrapped via torch.library.wrap_triton
    handles large, non-contiguous (channels_last) bfloat16 tensors correctly,
    addressing the issues seen in the original MaxPool2d bug (Issue 165297).
    """
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Reproduce the exact problematic configuration from the bug report
    N, C, H, W = 84, 64, 512, 960
    
    # Case 1: bfloat16 + channels_last
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=device)
    x = x.to(memory_format=torch.channels_last)

    print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Input stride: {x.stride()}")

    # Run the custom wrapped operation
    y = custom_max_pool(x)

    # Assertions to check for the bugs reported in the original issue
    assert not torch.isnan(y).any(), "Custom op produced NaNs on channels_last input"
    assert not torch.isinf(y).any(), "Custom op produced Infs on channels_last input"
    
    # Since the kernel is effectively an identity (max(x, x)), input and output should match
    assert torch.equal(x, y), "Output does not match input (kernel logic failed)"
    
    print("Test passed: wrap_triton kernel handled channels_last + bfloat16 correctly.")

if __name__ == "__main__":
    test_wrap_triton_channels_last()