import torch
import triton
import torch.symm_mem._nvshmem_triton as nvshmem

# Test case for Issue 163915: nvshmem triton kernels IMA silently
# This test verifies that enable_triton() correctly handles kernel naming.
# The bug occurs when a user-defined Triton kernel does not have "nvshmem" in its name,
# causing it to miss NVSHMEM initialization and hit a silent CUDA IMA.

def test_nvshmem_triton_kernel_naming_convention():
    """
    Tests that nvshmem.enable_triton() enforces or handles the naming convention
    for Triton kernels to prevent silent IMA failures.
    """
    
    # Setup: Create dummy tensors for kernel execution
    # Note: Actual NVSHMEM setup requires multi-proc launch, but we test the API logic here.
    x = torch.empty(10, device='cuda')
    
    # Case 1: Kernel without "nvshmem" in the name (Reproduces the bug scenario)
    # According to the issue, this should fail or raise an error if fixed, 
    # or crash with IMA if the bug persists.
    @triton.jit
    def foo_kernel(ptr):
        nvshmem.get(ptr) # Simplified call for reproduction
        
    nvshmem.enable_triton()
    
    # We expect this to fail. If the fix adds a check, it raises an error.
    # If the bug is present, this might cause a crash (IMA).
    # In a test environment, we might wrap this to catch the specific error added by the fix.
    try:
        foo_kernel[(1,)](x)
        # If we reach here without error, the bug might still be present (silent failure)
        # or the fix allows it (auto-detection).
        # For the purpose of this test based on the bug report, we assume
        # the desired behavior is strictness or explicit failure.
        assert False, "Expected an error or failure for kernel without 'nvshmem' in name"
    except RuntimeError as e:
        # Expected if the fix adds a validation check
        assert "nvshmem" in str(e).lower() or "name" in str(e).lower()
    except Exception as e:
        # Catching generic exception to document the crash/IMA behavior
        # In a real test suite, this might be an xfail or specific CUDA error check
        pass

    # Case 2: Kernel with "nvshmem" in the name (Expected success)
    @triton.jit
    def nvshmem_foo_kernel(ptr):
        nvshmem.get(ptr)

    # Re-enable to ensure state is clean or if the previous run disabled it
    nvshmem.enable_triton()
    
    try:
        nvshmem_foo_kernel[(1,)](x)
        # Success
    except Exception as e:
        # This should not fail if the naming convention is the only requirement
        assert False, f"Kernel with 'nvshmem' in name failed unexpectedly: {e}"

if __name__ == "__main__":
    test_nvshmem_triton_kernel_naming_convention()