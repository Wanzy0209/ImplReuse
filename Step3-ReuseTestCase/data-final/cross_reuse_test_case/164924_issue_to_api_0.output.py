import torch
import unittest

class TestIsinScalarCompile(unittest.TestCase):
    """
    Test case to verify torch.isin works correctly with scalar test_elements
    when using torch.compile (backend='inductor').
    
    This test is based on Issue ID: 164924.
    """
    
    def test_isin_scalar_test_elements_inductor(self):
        # Skip if CUDA is not available, as the original issue was CUDA-specific
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        torch.manual_seed(777)
        device = 'cuda'

        class IsinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Initialize a 1-D tensor and a 0-D tensor (scalar)
                self.x = torch.randint(-50, 50, (1,), dtype=torch.int64, device=device)
                self.y = torch.randint(-50, 50, (), dtype=torch.int64, device=device)

            def forward(self):
                # The bug occurs here when test_elements (self.y) is a scalar
                return torch.isin(self.x, self.y, assume_unique=False, invert=False)

        model = IsinModule()

        # 1. Run in Eager mode
        eager_output = model()
        
        # 2. Run in Compiled mode (Inductor)
        # The original bug report indicated a failure/crash or mismatch here
        compiled_model = torch.compile(model, backend='inductor')
        compiled_output = compiled_model()

        # 3. Verify that the outputs match
        self.assertTrue(
            torch.equal(eager_output, compiled_output),
            f"Eager and Compiled outputs differ.\nEager: {eager_output}\nCompiled: {compiled_output}"
        )

if __name__ == '__main__':
    unittest.main()