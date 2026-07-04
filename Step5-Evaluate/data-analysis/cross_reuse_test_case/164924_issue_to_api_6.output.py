import torch
import unittest

class TestIsinScalarCompile(unittest.TestCase):
    """
    Test case for Issue 164924: torch.isin fails for scalar test_elements in compile.
    This test preserves the original bug reproduction logic (scalar inputs) 
    and adapts the verification pattern to ensure consistency between eager and compiled modes.
    """

    @unittest.skipIf(not hasattr(torch, 'compile'), "torch.compile is not available (requires PyTorch 2.0+)")
    def test_isin_with_scalar_test_elements(self):
        # Setup inputs similar to the bug report
        # x is a 1D tensor
        x = torch.tensor([32], dtype=torch.int64)
        # y is a scalar tensor (0-d), which is the trigger for the bug
        y = torch.tensor(-21, dtype=torch.int64)

        # Define the operation logic
        def isin_logic(input_tensor, test_elements):
            return torch.isin(input_tensor, test_elements, assume_unique=False, invert=False)

        # 1. Eager Execution
        eager_result = isin_logic(x, y)
        
        # 2. Compiled Execution (Inductor)
        # We compile the function containing the isin call
        compiled_logic = torch.compile(isin_logic, backend='inductor')
        compiled_result = compiled_logic(x, y)

        # 3. Verification
        # Check that the compiled result matches the eager result
        self.assertTrue(torch.equal(eager_result, compiled_result), 
                        "Compiled output does not match eager output for scalar test_elements")
        
        # Specific assertion based on the values in the bug report
        # 32 is not in -21, so result should be False
        self.assertFalse(eager_result.item())
        self.assertFalse(compiled_result.item())

if __name__ == '__main__':
    unittest.main()