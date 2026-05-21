import torch
import unittest

# Translating the semantic of 'is_built_with_rocm' (environment/capability check)
# to ensure we are testing the specific environment where the bug occurs (CPU).
def is_target_environment_cpu():
    """
    Helper to check if we are running on CPU, similar to how 
    tf.test.is_built_with_rocm checks for ROCm support.
    """
    return not torch.cuda.is_available()

class TestTorchFullCompile(unittest.TestCase):
    def test_full_float64_compile_cpu(self):
        """
        Test that torch.full updates the fill_value correctly when 
        compiled with torch.compile for dtype=torch.float64 on CPU.
        
        Bug: The value argument seems to be fixed as the value in the first call.
        """
        # We explicitly force CPU execution to reproduce the bug condition
        device = torch.device('cpu')
        
        # If we wanted to strictly follow the 'skip if not capable' pattern 
        # of the similar API, we could skip if not CPU, but forcing device 
        # is more robust for testing the specific logic.
        # if not is_target_environment_cpu():
        #     self.skipTest("Test is only applicable on CPU")

        def func(x):
            return torch.full((2, ), x, dtype=torch.float64, device=device)

        # Compile the function
        func_jit = torch.compile(func)

        # First call with value 5.0
        x1 = torch.tensor(5.0, dtype=torch.float64, device=device)
        result1 = func_jit(x1)
        expected1 = torch.tensor([5.0, 5.0], dtype=torch.float64, device=device)
        self.assertTrue(torch.equal(result1, expected1), 
                        f"First call failed: expected {expected1}, got {result1}")

        # Second call with value 10.0
        # Bug: This returns [5., 5.] instead of [10., 10.]
        x2 = torch.tensor(10.0, dtype=torch.float64, device=device)
        result2 = func_jit(x2)
        expected2 = torch.tensor([10.0, 10.0], dtype=torch.float64, device=device)
        self.assertTrue(torch.equal(result2, expected2), 
                        f"Second call failed: expected {expected2}, got {result2}")

if __name__ == '__main__':
    unittest.main()