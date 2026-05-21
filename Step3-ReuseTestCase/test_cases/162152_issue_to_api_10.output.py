import unittest
import torch
import torch.nn as nn
from torch.nn import DataParallel

class SimpleModel(nn.Module):
    """
    A simple model used to reproduce the DataParallel custom backend issue.
    Structure matches the original bug report.
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TestDataParallelCustomBackend(unittest.TestCase):
    """
    Test case to verify torch.nn.DataParallel behavior.
    Note: Since the actual custom backend implementation (C++ binding) 
    is not available in this environment, we use CUDA as a proxy 
    to verify the DataParallel execution flow.
    """
    
    def test_dataparallel_forward_pass(self):
        # Check for CUDA availability as a proxy for the custom backend
        # In the original bug report, this was: torch.<mybackend>.is_available()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping test that requires multi-device setup.")

        # Check device count
        # In the original bug report, this was: torch.<mybackend>.device_count() > 1
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs to test DataParallel.")

        print(f"Detected {torch.cuda.device_count()} GPUs")

        # Initialize model
        model = SimpleModel()
        
        # Move model to device 0
        # In the original bug report, this was: model = model.<mybackend>()
        model = model.cuda()

        # Apply DataParallel
        # The bug report indicates issues with custom backends here.
        # We test with standard CUDA to ensure the logic flow is valid.
        model = DataParallel(model, device_ids=[0, 1])

        # Prepare input data
        batch_size = 20
        # In the original bug report, this was: torch.randn(...).<mybackend>()
        input_data = torch.randn(batch_size, 10).cuda()

        # Run forward pass
        try:
            output = model(input_data)
            print("DataParallel execution successful")
            
            # Assertions to verify output correctness
            self.assertIsNotNone(output)
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], 5) # output_size
            
        except Exception as e:
            self.fail(f"DataParallel failed with error: {e}")

if __name__ == '__main__':
    unittest.main()