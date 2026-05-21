import torch
import torch.nn as nn

def test_embeddingbag_2d_include_last_offset():
    """
    Test case for Issue 167974: EmbeddingBag sum has incorrect offsets 
    when input is 2D and include_last_offset is set to True.
    
    This test leverages the pattern of checking tensor dimensions (similar to 
    tf.keras.ops.ndim) to ensure the input is 2D before verifying the offset logic.
    """
    # Initialize EmbeddingBag with include_last_offset=True
    embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
    
    # Create a 2D input tensor (2 bags, 4 indices each)
    # This matches the structure found in the bug report and the similar API usage
    input_indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    
    # Leverage the dimension checking pattern from the similar API (tf.keras.ops.ndim)
    # to explicitly verify the input rank before processing.
    # tf.keras.ops.ndim returns the rank; here we use .dim() for PyTorch.
    input_rank = input_indices.dim()
    assert input_rank == 2, f"Input tensor must be 2D to reproduce the bug, got rank {input_rank}"

    # Execute the forward pass
    output = embedding_sum(input_indices)

    # Verify the fix:
    # Bug behavior: Offsets generated as [0, 4]. The backend interprets this as 1 bag (indices 0-4).
    #               Output shape would be (1, 3).
    # Correct behavior: Offsets generated as [0, 4, 8]. The backend interprets this as 2 bags (indices 0-4, 4-8).
    #                   Output shape should be (2, 3).
    expected_num_bags = input_indices.size(0)
    actual_num_bags = output.size(0)

    assert actual_num_bags == expected_num_bags, (
        f"Expected output to have {expected_num_bags} bags, but got {actual_num_bags}. "
        "This indicates that the offsets were likely generated incorrectly "
        "(missing the last offset) for 2D input with include_last_offset=True."
    )
    
    # Also verify the embedding dimension is preserved
    assert output.size(1) == 3

if __name__ == "__main__":
    test_embeddingbag_2d_include_last_offset()
    print("Test passed successfully.")