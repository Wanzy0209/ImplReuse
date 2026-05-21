import torch
import unittest

class TestAbsCPUFallback(unittest.TestCase):
    def test_abs_functional_shape(self):
        """
        Test that torch.abs returns a tensor with the correct shape
        when dispatched via CompositeExplicitAutograd through cpu_fallback.
        
        Bug: Operations like abs return empty tensors (shape [0]) when run 
        through cpu_fallback on custom backends.
        """
        # Check if a custom backend is registered to privateuse1
        if not torch.has_privateuse1:
            self.skipTest("privateuse1 backend not available")

        # Create a tensor on the custom device
        input_tensor = torch.randn(4, 4, device='privateuse1')
        
        # Perform the operation
        result = torch.abs(input_tensor)

        # Verify the shape matches the input (Bug ID 165549 check)
        self.assertEqual(result.shape, input_tensor.shape, 
                         f"Shape mismatch: expected {input_tensor.shape}, got {result.shape}")
        
        # Verify the device is preserved
        self.assertEqual(result.device, input_tensor.device)

        # Verify the values are non-negative (sanity check for abs)
        self.assertTrue(torch.all(result >= 0))

    def test_abs_in_place_and_out_variants(self):
        """
        Test that in-place and out= variants work correctly as mentioned 
        in the bug report (these were not broken, but good to verify).
        """
        if not torch.has_privateuse1:
            self.skipTest("privateuse1 backend not available")

        t = torch.randn(4, 4, device='privateuse1')

        # Test in-place
        t_inplace = t.clone()
        t_inplace.abs_()
        self.assertEqual(t_inplace.shape, t.shape)

        # Test out=
        out_tensor = torch.empty_like(t)
        torch.abs(t, out=out_tensor)
        self.assertEqual(out_tensor.shape, t.shape)

if __name__ == '__main__':
    unittest.main()