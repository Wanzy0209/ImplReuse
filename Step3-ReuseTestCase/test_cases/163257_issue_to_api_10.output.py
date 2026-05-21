import unittest
import torch

# Check for the presence of the RMM library used in the bug report
try:
    import rmm
    from rmm.allocators.torch import rmm_torch_allocator
    HAS_RMM = True
except ImportError:
    HAS_RMM = False

@unittest.skipIf(not HAS_RMM, "RMM library is not installed, skipping pluggable allocator test")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestPluggableAllocatorCompile(unittest.TestCase):
    """
    Test case to verify that torch.compile works with pluggable device allocators.
    This addresses Issue #163257 where checkPoolLiveAllocations was not supported.
    """

    def test_compile_with_rmm_allocator(self):
        """
        Reproduces the logic from the bug report:
        1. Initialize RMM pool allocator.
        2. Set it as the current PyTorch allocator.
        3. Attempt to compile a simple function.
        
        Expected behavior (post-fix): The code should run without raising
        RuntimeError regarding checkPoolLiveAllocations.
        """
        # Store original allocator to restore after test
        # Note: PyTorch API for getting the allocator might vary by version,
        # but change_current_allocator is the focus here.
        
        try:
            # Step 1 & 2: Setup RMM allocator (from bug report)
            rmm.reinitialize(pool_allocator=True)
            torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

            # Step 3: Define a simple function to compile
            def simple_model(x):
                return x + 1

            # Attempt to compile
            # In the bug report, this raised:
            # RuntimeError: pluggable does not yet support checkPoolLiveAllocations.
            compiled_model = torch.compile(simple_model)
            
            # Verify execution
            input_tensor = torch.randn(10).cuda()
            output = compiled_model(input_tensor)
            
            # Basic assertion to ensure execution happened
            self.assertEqual(output.shape, input_tensor.shape)
            self.assertTrue(torch.allclose(output, input_tensor + 1))

        except RuntimeError as e:
            if "checkPoolLiveAllocations" in str(e):
                self.fail(f"Bug reproduced: {e}")
            else:
                # Re-raise other RuntimeErrors
                raise
        finally:
            # Cleanup: Reinitialize RMM to default or restore PyTorch allocator
            # to avoid affecting other tests. 
            # Ideally, we restore the previous allocator, but simply reinitializing 
            # RMM to a fresh state or resetting to PyTorch's default is a safe cleanup
            # for this specific test context.
            rmm.reinitialize(pool_allocator=False) 
            # Note: Restoring the exact previous allocator object is not always 
            # straightforward via the public API without saving it beforehand.
            # We assume the environment handles allocator reset or this test runs in isolation.

if __name__ == '__main__':
    unittest.main()