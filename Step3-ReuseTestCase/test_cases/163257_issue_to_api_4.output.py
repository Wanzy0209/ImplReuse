import torch
import rmm
from rmm.allocators.torch import rmm_torch_allocator

def test_pluggable_allocator_with_compile():
    """
    Test case to verify that torch.compile works with a pluggable allocator (RMM).
    This test reproduces the logic from Issue 163257 and leverages the pattern
    of tf.math.zero_fraction to verify memory integrity.
    """
    
    # 1. Setup the pluggable allocator (Original Bug Reproduction Logic)
    # This mimics the user's need to share memory pools between PyTorch and RAPIDS.
    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # 2. Define a simple model to be compiled
    def simple_model(x):
        # A simple operation to ensure kernels are launched
        return x * 2 + 1

    # 3. Compile the model
    # In the bug report, this step raised:
    # RuntimeError: pluggable does not yet support checkPoolLiveAllocations.
    # We expect this to succeed now.
    try:
        compiled_model = torch.compile(simple_model)
    except RuntimeError as e:
        if "checkPoolLiveAllocations" in str(e):
            print("FAIL: The pluggable allocator does not support checkPoolLiveAllocations.")
            raise
        else:
            raise

    # 4. Run the compiled model
    # Using random data to ensure we are actually writing to memory allocated by RMM
    input_tensor = torch.randn(1024, 1024, device='cuda')
    output_tensor = compiled_model(input_tensor)

    # 5. Leverage Similar API (tf.math.zero_fraction) logic
    # We translate the semantics of tf.math.zero_fraction to PyTorch to verify
    # that the allocator did not corrupt memory (e.g., by zeroing it out incorrectly).
    # tf.math.zero_fraction returns the fraction of zeros in value.
    
    def get_zero_fraction(tensor):
        """
        PyTorch equivalent of tf.math.zero_fraction.
        Returns the fraction of zeros in the tensor.
        """
        if tensor.numel() == 0:
            return float('nan')
        
        # Count zeros
        num_zeros = (tensor == 0).sum().item()
        total_elements = tensor.numel()
        
        return num_zeros / total_elements

    zero_frac = get_zero_fraction(output_tensor)

    # 6. Assertions
    # Since input is random normal (mean 0, std 1) and we do x*2 + 1,
    # the probability of output being exactly 0 is extremely low.
    # A high zero fraction would indicate memory corruption or allocator failure.
    assert zero_frac < 0.001, (
        f"Memory integrity check failed. Zero fraction is {zero_frac:.4f}, "
        "which suggests the pluggable allocator might be returning zeroed memory."
    )
    
    # Also verify the computation is correct
    expected = input_tensor * 2 + 1
    assert torch.allclose(output_tensor, expected), "Computation result mismatch."

    print("Test Passed: Pluggable allocator supports checkPoolLiveAllocations and memory integrity is verified.")

if __name__ == "__main__":
    test_pluggable_allocator_with_compile()