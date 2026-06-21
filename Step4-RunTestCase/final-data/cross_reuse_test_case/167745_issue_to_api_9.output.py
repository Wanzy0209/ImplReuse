import torch
import unittest

class TestMemPoolWithSeed(unittest.TestCase):
    def test_temporary_mempool_with_initial_seed(self):
        """
        Test case based on Issue 167745.
        Reproduces the failing pattern of temporary MemPool objects
        and leverages torch.cuda.initial_seed to verify CUDA state consistency.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Assuming make_custom_pool is available in the environment as per the bug report.
        # This helper is typically defined in the external repo associated with the issue.
        try:
            # In a real test environment, this would import the actual helper.
            # For this snippet, we assume it exists or the test is skipped.
            from test_utils import make_custom_pool
        except ImportError:
            self.skipTest("make_custom_pool helper not found")

        # Reproduce Test 1 pattern from the bug report:
        # Creating a pool and using it immediately in a context manager
        pool1 = make_custom_pool(1)

        with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
            # Leverage the similar API: torch.cuda.initial_seed
            # This checks if the CUDA random state (which involves lazy initialization)
            # is accessible and valid inside the custom MemPool context.
            seed = torch.cuda.initial_seed()
            self.assertIsInstance(seed, int, "Initial seed should be an integer")

            # Original bug reproduction logic: allocation inside the pool
            x1 = torch.randn(8, device="cuda")
            self.assertEqual(x1.shape, (8,))
            self.assertEqual(x1.device.type, "cuda")

        # Cleanup as in the original report
        del x1

if __name__ == "__main__":
    unittest.main()