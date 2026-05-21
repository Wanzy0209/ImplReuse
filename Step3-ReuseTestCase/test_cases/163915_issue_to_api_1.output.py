import torch
import triton
import torch.symm_mem._nvshmem_triton as nvshmem
import unittest

class TestNvshmemTritonInitialization(unittest.TestCase):
    """
    Test case for Issue 163915: nvshmem triton kernels IMA silently.
    
    This test verifies the behavior of torch.symm_mem._nvshmem_triton.enable_triton
    regarding kernel naming conventions. It leverages the initialization pattern
    similar to tf.compat.v1.global_variables_initializer, where an explicit
    initialization step is required before kernel execution.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_enable_triton_kernel_naming_validation(self):
        """
        Reproduces the scenario where a kernel does not have 'nvshmem' in its name.
        
        The bug report indicates that missing this requirement leads to a silent IMA.
        This test asserts that the API now handles this explicitly, either by
        raising a clear error or by successfully initializing regardless of the name
        (depending on the specific fix implementation). Here we assume the fix
        enforces a check to prevent silent failures.
        """
        # Define a kernel without 'nvshmem' in the name (Reproducing the bug condition)
        @triton.jit
        def foo(x):
            nvshmem.get(x)

        # Initialize NVSHMEM for Triton
        # This mirrors the usage of tf.compat.v1.global_variables_initializer
        # where global state is prepared before execution.
        nvshmem.enable_triton()

        # Prepare input
        x = torch.empty(1, device='cuda')

        # Attempt to run the kernel
        # We expect a RuntimeError to be raised if the naming convention is enforced
        # to prevent the silent IMA described in the bug.
        with self.assertRaises(RuntimeError) as context:
            foo[(1,)](x)
        
        # Verify the error message is related to the naming convention or initialization
        self.assertTrue("nvshmem" in str(context.exception).lower() or "init" in str(context.exception).lower())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_enable_triton_with_correct_naming(self):
        """
        Tests the positive case where the kernel name includes 'nvshmem'.
        This should execute without raising a naming-related error.
        """
        # Define a kernel with 'nvshmem' in the name
        @triton.jit
        def bar_nvshmem(x):
            nvshmem.get(x)

        nvshmem.enable_triton()
        x = torch.empty(1, device='cuda')

        # This should run without raising a naming convention error
        # (Note: It may still fail if actual NVSHMEM resources are not configured,
        # but it should not fail due to the naming check specifically).
        try:
            bar_nvshmem[(1,)](x)
        except RuntimeError as e:
            # If it fails, ensure it's not the naming convention error
            self.assertFalse("nvshmem" in str(e).lower() and "name" in str(e).lower())

if __name__ == '__main__':
    unittest.main()