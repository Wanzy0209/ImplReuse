import torch
import sys

def test_compile_with_custom_allocator_and_bucketize():
    """
    Test case for Issue 163257: Support checkPoolLiveAllocations with pluggable device.
    
    This test combines the 'Original API Under Test' (torch.cuda.memory.change_current_allocator)
    with the 'Similar API' (torch.bucketize) to verify that torch.compile handles custom 
    allocators correctly when lowering operations that require backend feature checks.
    """
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    # Check for RMM (Rapids Memory Manager) availability
    # The bug report specifically uses RMM as the pluggable allocator.
    try:
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator
    except ImportError:
        print("SKIP: RMM (rapids-memory-manager) not installed")
        return

    # 1. Setup the custom allocator (Reproducing the bug environment)
    # This corresponds to the "Original API Under Test".
    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # 2. Define a function using the "Similar API": torch.bucketize
    # The implementation of torch.bucketize in torch._inductor.lowering checks
    # for BackendFeature.BUCKETIZE before proceeding. We use this operation
    # to ensure the compilation pipeline respects feature availability and
    # allocator constraints simultaneously.
    def bucketize_fn(input_tensor, boundaries):
        return torch.bucketize(input_tensor, boundaries)

    # 3. Compile the function
    # In the original bug, this step would raise:
    # RuntimeError: pluggable does not yet support checkPoolLiveAllocations.
    # This test verifies that the fix allows compilation to proceed or fail gracefully.
    try:
        compiled_fn = torch.compile(bucketize_fn)
    except RuntimeError as e:
        if "checkPoolLiveAllocations" in str(e):
            print(f"FAIL: The original bug is present - {e}")
            raise
        else:
            # Re-raise other RuntimeErrors unrelated to the specific bug
            raise

    # 4. Execute the compiled function with test data
    input_tensor = torch.tensor([0.2, 0.5, 0.8, 1.2], device='cuda')
    boundaries = torch.tensor([0.0, 0.5, 1.0], device='cuda')

    result = compiled_fn(input_tensor, boundaries)
    
    # 5. Verify correctness against the eager execution
    expected = bucketize_fn(input_tensor, boundaries)
    
    assert torch.equal(result, expected), "Mismatch between compiled and eager execution"
    print("PASS: torch.compile works with custom allocator and bucketize operation.")

if __name__ == "__main__":
    test_compile_with_custom_allocator_and_bucketize()