import torch
import torch.nn as nn

def test_embedding_bag_2d_include_last_offset():
    """
    Test case for Issue 167974:
    Verifies that EmbeddingBag handles 2D input correctly when include_last_offset is True.
    
    The bug report indicates that when input is 2D, the automatically generated offsets
    do not include the last element (size of indices) as required by the documentation
    when include_last_offset=True.
    """
    # Initialize EmbeddingBag with include_last_offset=True
    embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
    
    # Create a 2D input tensor
    # Shape: (2, 4) -> 2 examples in the batch, 4 indices each
    input_indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    
    # Run the forward pass
    # This triggers the automatic offset generation logic in torch.nn.functional.py
    output = embedding_sum(input_indices)
    
    # Assertions to verify the behavior
    # 1. Check output shape. For 2D input of shape (B, N), output should be (B, EmbeddingDim)
    assert output.shape == (2, 3), f"Expected output shape (2, 3), but got {output.shape}"
    
    # 2. Verify the output values are correct (sum of embeddings)
    # We manually calculate the expected sum to ensure the forward pass logic
    # remains correct regardless of the internal offset representation.
    weights = embedding_sum.weight
    
    # Row 0: indices [1, 2, 4, 5]
    expected_row_0 = weights[1] + weights[2] + weights[4] + weights[5]
    
    # Row 1: indices [4, 3, 2, 9]
    expected_row_1 = weights[4] + weights[3] + weights[2] + weights[9]
    
    expected_output = torch.stack([expected_row_0, expected_row_1])
    
    assert torch.allclose(output, expected_output), "Output values do not match the expected sum of embeddings"
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_embedding_bag_2d_include_last_offset()