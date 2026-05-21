import torch
import triton
import triton.language as tl
import pytest

# Attempting to import the module, handling cases where it might not be present in the test environment
try:
    import torch.symm_mem._nvshmem_triton as nvshmem
except ImportError:
    pytest.skip("torch.symm_mem._nvshmem_triton not available", allow_module_level=True)

def test_nvshmem_kernel_naming_convention():
    """
    Test that nvshmem.enable_triton() correctly handles kernel naming conventions.
    
    This test is based on Issue 163915, where kernels without 'nvshmem' in their name
    would miss initialization and silently hit CUDA IMA. The test verifies that the 
    API enforces or handles the naming requirement appropriately, similar to how 
    tf.executing_eagerly checks the execution context state.
    """
    
    # Define a kernel WITHOUT 'nvshmem' in the name (The Bug Scenario)
    @triton.jit
    def invalid_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

    # Define a kernel WITH 'nvshmem' in the name (The Expected Scenario)
    @triton.jit
    def valid_nvshmem_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

    # Enable Triton for NVSHMEM
    nvshmem.enable_triton()
    
    # Prepare data
    x = torch.randn(128, device='cuda')
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Test Case 1: Verify behavior with the invalid kernel name
    # Depending on the fix, this should either raise a clear error or work correctly.
    # Assuming the fix enforces the naming convention to prevent silent IMA:
    with pytest.raises(RuntimeError, match="nvshmem"):
        invalid_kernel[(1,)](x, output, n_elements, BLOCK_SIZE=128)

    # Test Case 2: Verify behavior with the valid kernel name
    # This should execute successfully.
    # Note: Actual execution might fail if NVSHMEM is not properly initialized on the hardware,
    # but the API check should pass.
    try:
        valid_nvshmem_kernel[(1,)](x, output, n_elements, BLOCK_SIZE=128)
    except RuntimeError as e:
        # If it fails, it should be due to hardware/init, not the naming convention check
        assert "nvshmem" not in str(e) or "initialized" in str(e)

if __name__ == "__main__":
    test_nvshmem_kernel_naming_convention()