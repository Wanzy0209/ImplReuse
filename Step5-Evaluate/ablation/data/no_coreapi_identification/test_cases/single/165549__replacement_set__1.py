import torch
import unittest

class TestCompositeExplicitAutogradFallback(unittest.TestCase):
    def test_abs_with_var_mean_and_cos(self):
        """
        Test case for Issue 165549 adapted with API replacements.
        Replacements:
        - torch.randn -> torch.var_mean
        - torch.neg -> torch.cos
        """
        # Check if privateuse1 is available (requires a custom backend to be registered)
        if not torch.has_privateuse1:
            self.skipTest("PrivateUse1 backend not available")

        # Replacement for torch.randn: torch.var_mean
        # We need an input tensor to calculate variance/mean. 
        # We use torch.ones to create the initial data on the custom device.
        input_data = torch.ones(4, 4, device='privateuse1')
        
        # torch.var_mean returns a tuple (variance, mean). 
        # We use the variance tensor as 't' for the test.
        # We compute along dim=0 to get a non-scalar tensor (shape [4]) to test shape preservation.
        t, _ = torch.var_mean(input_data, dim=0)

        # This fails - returns tensor with shape [0] in the buggy version
        result = torch.abs(t)
        self.assertEqual(result.shape, t.shape, 
                         f"Expected shape {t.shape}, but got {result.shape}. "
                         "Bug: CompositeExplicitAutograd ops return empty tensors via cpu_fallback.")

        # But these work correctly
        # 1. Out variant
        pre_allocated_tensor = torch.empty_like(t)
        torch.abs(t, out=pre_allocated_tensor)
        self.assertEqual(pre_allocated_tensor.shape, t.shape)

        # 2. In-place variant
        t_copy = t.clone()
        t_copy.abs_()
        self.assertEqual(t_copy.shape, t.shape)

        # Replacement for torch.neg: torch.cos
        # Original: torch.neg(t) works (uses structured_delegate)
        # New: torch.cos(t) should work
        cos_result = torch.cos(t)
        self.assertEqual(cos_result.shape, t.shape)

if __name__ == '__main__':
    unittest.main()