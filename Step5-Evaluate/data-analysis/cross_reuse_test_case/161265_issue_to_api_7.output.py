import torch
import unittest

class TestMPSLargeTensorFill(unittest.TestCase):
    """
    Test case for Issue 161265: [MPS] On MacOS-26 torch.full fails for 4+Gb tensors.
    
    This test preserves the original bug reproduction logic while leveraging the 
    availability check pattern similar to torch.backends.nnpack.is_available.
    """
    
    def test_mps_large_tensor_fill(self):
        # Leveraging the pattern of checking backend availability, 
        # similar to torch.backends.nnpack.is_available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available")

        # Original bug reproduction logic:
        # Create a tensor larger than 4GB (2 * (2^31 + 5) bytes)
        # This triggers the Metal fillBuffer issue where values beyond 4GB were not set.
        a = torch.ones(2, (1 << 31) + 5, dtype=torch.int8, device='mps')

        # Assertions to verify the fix.
        # The bug report showed a[1, -2] was 0 instead of 1.
        self.assertEqual(a[1, -2].item(), 1, 
                         "Element at a[1, -2] should be 1, but fillBuffer failed for >4GB range")
        
        # Verify the slice as well to ensure consistency across the tensor
        slice_result = a[:, -2]
        self.assertTrue(torch.all(slice_result == 1), 
                        "All elements in the slice a[:, -2] should be 1")

if __name__ == '__main__':
    unittest.main()