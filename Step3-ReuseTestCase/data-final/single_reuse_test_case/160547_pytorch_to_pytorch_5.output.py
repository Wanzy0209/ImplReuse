import torch
import torch.nn as nn
from collections import namedtuple

def test_dataparallel_namedtuple():
    """
    Test case to verify torch.nn.DataParallel handles NamedTuple inputs correctly.
    Adapted from Issue 160547 regarding torch.export.export behavior with NamedTuples.
    """
    # Define the NamedTuple
    Point = namedtuple('Point', 'x y')
    
    # Define a simple Module
    class M(nn.Module):
        def forward(self, x, y):
            return x + y 
    
    # Create input tensors and wrap in NamedTuple
    x = torch.ones(3)
    y = torch.ones(3)
    inp = Point(x, y)
    
    # Setup DataParallel
    # Note: We use CPU device to ensure the test is runnable without requiring GPUs.
    # DataParallel logic for input handling remains the same regardless of device.
    device = torch.device('cpu')
    model = M().to(device)
    dp = nn.DataParallel(model, device_ids=[device])
    
    # Test 1: Passing the NamedTuple directly
    # DataParallel should recognize the namedtuple as a tuple, scatter its elements (x and y),
    # and pass them as positional arguments to the module's forward method.
    try:
        output = dp(inp)
        expected = x + y
        assert torch.equal(output, expected), "DataParallel failed to handle NamedTuple input directly"
        print("Test passed: DataParallel handles NamedTuple input directly")
    except Exception as e:
        print(f"Test failed: DataParallel failed with NamedTuple input: {e}")
        raise

    # Test 2: Unpacking the NamedTuple (Standard usage)
    # This verifies that the basic setup works as expected.
    try:
        output_unpacked = dp(*inp)
        assert torch.equal(output_unpacked, expected), "DataParallel failed with unpacked NamedTuple"
        print("Test passed: DataParallel handles unpacked NamedTuple")
    except Exception as e:
        print(f"Test failed: DataParallel failed with unpacked NamedTuple: {e}")
        raise

if __name__ == "__main__":
    test_dataparallel_namedtuple()