import torch
import unittest

class TestRepeatInterleaveLayout(unittest.TestCase):
    def test_repeat_interleave_with_non_contiguous_input(self):
        """
        Adapted from Issue 164491: _scaled_mm and _int_mm slow/errors with row-major rhs.
        
        The original bug highlights performance and correctness issues when matrix 
        multiplication APIs encounter specific memory layouts (row-major vs column-major).
        
        This test verifies that torch.repeat_interleave handles non-contiguous 
        (column-major view) inputs correctly and robustly, ensuring it does not 
        raise errors or produce incorrect results due to tensor strides.
        """
        # Create a standard row-major matrix
        x = torch.arange(12, dtype=torch.float32).view(3, 4)
        
        # Create a non-contiguous (column-major view) version by transposing
        # This mimics the scenario where weights are transposed for backward/forward passes
        x_t = x.t()
        
        # Verify the tensor is indeed non-contiguous
        self.assertFalse(x_t.is_contiguous())

        # Test repeat_interleave on the non-contiguous tensor along dim 0
        repeats = 2
        result = torch.repeat_interleave(x_t, repeats=repeats, dim=0)
        
        # Calculate expected result by forcing contiguity first
        # If the API handles strides correctly, these should be identical
        expected = torch.repeat_interleave(x_t.contiguous(), repeats=repeats, dim=0)
        
        self.assertTrue(torch.equal(result, expected))
        
        # Test with a different dimension to ensure stride handling is general
        result_dim1 = torch.repeat_interleave(x_t, repeats=3, dim=1)
        expected_dim1 = torch.repeat_interleave(x_t.contiguous(), repeats=3, dim=1)
        
        self.assertTrue(torch.equal(result_dim1, expected_dim1))

    def test_repeat_interleave_with_strided_repeats(self):
        """
        Test that repeat_interleave handles non-contiguous 'repeats' tensors correctly.
        """
        x = torch.tensor([1, 2, 3, 4])
        
        # Create a non-contiguous repeats tensor
        repeats_full = torch.tensor([1, 2, 3, 4])
        repeats_strided = repeats_full[::2] # [1, 3]
        
        self.assertFalse(repeats_strided.is_contiguous())
        
        result = torch.repeat_interleave(x, repeats_strided)
        expected = torch.tensor([1, 3, 3, 3])
        
        self.assertTrue(torch.equal(result, expected))

if __name__ == '__main__':
    unittest.main()