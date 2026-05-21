import torch
import sys

# Check for RMM availability as it is required to reproduce the specific pluggable allocator issue
try:
    import rmm
    from rmm.allocators.torch import rmm_torch_allocator
except ImportError:
    print("Skipping test: RMM library is not installed.")
    sys.exit(0)

def test_compile_with_pluggable_allocator():
    """
    Test case to verify that torch.compile works correctly when using a 
    pluggable memory allocator (RMM). 
    
    This test leverages the relationship between the allocator setup 
    (torch.cuda.memory.change_current_allocator) and the allocation API 
    (torch.empty) which is lowered by the inductor.
    """
    
    # 1. Setup the custom allocator (Original API Under Test)
    # This mimics the user's setup in the bug report.
    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # 2. Define a workload using torch.empty (Similar API)
    # The bug report indicates that the error occurs during compilation/execution,
    # likely when the inductor lowers operations to memory allocations like 'empty'.
    def simple_kernel(x):
        # Explicitly using torch.empty to trigger the allocation path
        # that interacts with the pluggable allocator.
        y = torch.empty_like(x)
        return y + x

    # 3. Compile the function
    # The error "pluggable does not yet support checkPoolLiveAllocations"
    # typically occurs here or during the first invocation.
    try:
        compiled_fn = torch.compile(simple_kernel)
    except RuntimeError as e:
        if "checkPoolLiveAllocations" in str(e):
            print(f"Bug Reproduced: {e}")
            # Re-raise to indicate test failure due to the known bug
            raise
        else:
            raise

    # 4. Execute the compiled function
    input_tensor = torch.randn(10, 10, device='cuda')
    
    try:
        output = compiled_fn(input_tensor)
        
        # Verify correctness
        expected = input_tensor + input_tensor
        assert torch.allclose(output, expected), "Output mismatch"
        
        print("Test Passed: torch.compile is compatible with the pluggable allocator.")
        
    except RuntimeError as e:
        if "checkPoolLiveAllocations" in str(e):
            print(f"Bug Reproduced during execution: {e}")
            raise
        else:
            raise

if __name__ == "__main__":
    test_compile_with_pluggable_allocator()