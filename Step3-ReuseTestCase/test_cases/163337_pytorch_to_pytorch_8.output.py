import torch

# Adapted test case for torch.nan_to_num
# The original issue involved float to __half (float16) conversion issues during compilation.
# This test verifies the runtime behavior of torch.nan_to_num with float16 tensors,
# ensuring correct handling of special floating-point values in half precision.

def test_nan_to_num():
    # Create a tensor containing NaN, Inf, -Inf, and normal values
    # Using torch.float16 (half) to align with the __half context in the bug report
    input_tensor = torch.tensor([float('nan'), float('inf'), float('-inf'), 1.5, -1.5], dtype=torch.float16)
    
    # Call torch.nan_to_num with specific replacement values
    # nan=0.0, posinf=1.0, neginf=-1.0
    result = torch.nan_to_num(input_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Define the expected output tensor
    expected = torch.tensor([0.0, 1.0, -1.0, 1.5, -1.5], dtype=torch.float16)
    
    # Assert that the result matches the expected output
    assert torch.equal(result, expected), f"Test failed: Expected {expected}, got {result}"
    
    print("torch.nan_to_num test passed.")

if __name__ == "__main__":
    test_nan_to_num()