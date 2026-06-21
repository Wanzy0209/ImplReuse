import torch
import torch.nn as nn
from collections import namedtuple

def test_dataparallel_namedtuple():
    """
    Test case to verify torch.nn.DataParallel handles NamedTuple inputs correctly.
    Adapted from Issue 160547 regarding torch.export.export behavior with NamedTuples.
    """
    # DataParallel requires CUDA devices. Skip if not available.
    if not torch.cuda.is_available():
        print("Test skipped: CUDA is not available. DataParallel requires GPU devices.")
        return

    # Define the NamedTuple
    Point = namedtuple('Point', 'x y')
    
    # Define a simple Module
    class M(nn.Module):
        def forward(self, x, y):
            return x + y 
    
    # Setup Device
    # DataParallel requires non-CPU device_ids for parallel processing.
    device = torch.device('cuda:0')

    # Create input tensors and wrap in NamedTuple
    # Inputs must be on the same device as the model for DataParallel to work correctly.
    x = torch.ones(3).to(device)
    y = torch.ones(3).to(device)
    inp = Point(x, y)
    
    # Setup DataParallel
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