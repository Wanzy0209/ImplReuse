import torch
import torch.nn as nn

def test_embeddingbag_2d_include_last_offset():
    """
    Test case for Issue 167974:
    EmbeddingBag sum has incorrect offsets when input is 2D and include_last_offset is set to True.
    
    The bug causes the automatically generated offsets for a 2D input to ignore the 
    'include_last_offset' flag, resulting in offsets [0, N] instead of [0, N, 2N, ...].
    This test verifies that the output matches the expected behavior when offsets are 
    correctly generated.
    """
    # Initialize EmbeddingBag with include_last_offset=True
    # We use a fixed seed for reproducibility
    torch.manual_seed(42)
    embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
    
    # 2D input tensor: 2 bags, each of size 4
    input_2d = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    
    # --- Ground Truth Calculation ---
    # Flatten the input to 1D
    input_1d = input_2d.view(-1)
    
    # Manually construct the correct offsets for include_last_offset=True
    # For 2 rows of size 4, offsets should be [0, 4, 8]
    # The last element (8) is the size of the indices (input_1d.size(0))
    expected_offsets = torch.tensor([0, 4, 8], dtype=torch.long)
    
    # Compute the expected output using the manually constructed offsets
    expected_output = embedding_sum(input_1d, offsets=expected_offsets)
    
    # --- Test Case Execution ---
    # Call EmbeddingBag with 2D input. This triggers the automatic offset generation.
    # If the bug is present, it generates [0, 4] internally, leading to incorrect results.
    actual_output = embedding_sum(input_2d)
    
    # --- Assertions ---
    # 1. Check shape: With 2 input rows, we expect 2 output bags.
    # If the bug occurs (offsets [0, 4]), the output might be shape (1, 3) or incorrect.
    assert actual_output.shape == expected_output.shape, \
        f"Shape mismatch: expected {expected_output.shape}, got {actual_output.shape}"
        
    # 2. Check values: The result should match the ground truth.
    assert torch.allclose(actual_output, expected_output), \
        f"Output mismatch:\nExpected: {expected_output}\nActual: {actual_output}"

    print("Test passed: EmbeddingBag correctly handles 2D input with include_last_offset=True.")

if __name__ == "__main__":
    test_embeddingbag_2d_include_last_offset()