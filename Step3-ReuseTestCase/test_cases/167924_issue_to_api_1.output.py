import torch
import unittest

class TestMPSRepeatInterleave(unittest.TestCase):
    """
    Test case for Issue 167924: Crash on MPS when using repeat_interleave with sliced tensor.
    
    This test preserves the original bug reproduction logic. It mirrors the pattern 
    found in the similar API (tf.compat.v1.local_variables_initializer) by checking 
    the execution context (MPS availability) before performing the operation.
    """

    def test_repeat_interleave_with_non_prefix_slice(self):
        # Check for MPS availability, similar to context.executing_eagerly() check
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available, skipping MPS-specific test.")

        # Reproduce the bug scenario
        counts = torch.tensor([0, 1, 0], device="mps")
        data = torch.arange(2, device="mps")
        
        # The bug is triggered by slicing the counts tensor to a non-prefix range
        # before passing it to repeat_interleave.
        sliced_counts = counts[1:3]
        
        # This operation caused a segfault on MPS in the reported version.
        # We assert that it runs successfully and produces the correct output.
        result = data.repeat_interleave(sliced_counts, dim=0)
        
        # Verify the result matches the expected behavior
        # counts[1:3] is [1, 0]. 
        # data[0] (0) repeats 1 time -> [0]
        # data[1] (1) repeats 0 times -> []
        # Expected result: [0]
        expected = torch.tensor([0], device="mps")
        self.assertTrue(torch.equal(result, expected))

if __name__ == "__main__":
    unittest.main()