import torch
import unittest

class TestInductorUIntNegation(unittest.TestCase):
    """
    Test case for Issue 161763: neg+add computation including uint tensor is incorrect under inductor.
    
    This test verifies that torch.compile (using the inductor backend) correctly handles
    the negation of unsigned integer tensors and their subsequent addition to floating point tensors.
    """
    
    def test_neg_add_uint_tensor(self):
        def model(x):
            # Create a uint8 tensor
            c = torch.tensor(7, dtype=torch.uint8)
            # Return:
            # 1. c + x (uint + float)
            # 2. neg(c) (negation of uint, wraps around)
            # 3. neg(c) + x (wrapped uint + float)
            return c + x, torch.neg(c), torch.neg(c) + x

        torch.manual_seed(0)
        x = torch.randn(2, 2, dtype=torch.float32)
        
        # Eager execution
        eager_res = model(x)
        
        # Compiled execution
        compiled_model = torch.compile(model)
        compiled_res = compiled_model(x)
        
        # Assert that compiled results match eager results
        # Check simple addition
        torch.testing.assert_close(eager_res[0], compiled_res[0])
        
        # Check negation
        torch.testing.assert_close(eager_res[1], compiled_res[1])
        
        # Check negation + addition
        # This is the specific case that failed in the bug report.
        # Eager: 249 + x (since neg(7) in uint8 is 249)
        # Buggy Compiled: -7 + x (incorrectly treating negation as signed)
        torch.testing.assert_close(eager_res[2], compiled_res[2])

if __name__ == '__main__':
    unittest.main()