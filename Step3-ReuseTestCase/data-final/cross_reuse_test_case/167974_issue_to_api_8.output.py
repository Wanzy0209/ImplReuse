import torch
import torch.nn as nn

def test_embeddingbag_2d_input_with_include_last_offset():
    """
    Test case for Issue #167974.
    
    Verifies that torch.nn.EmbeddingBag with include_last_offset=True correctly
    generates offsets when the input is a 2D tensor.
    
    Bug Description:
    When input is 2D and include_last_offset is True, the framework should generate
    offsets equivalent to [0, n, 2n, ...] where the last element is the total size
    of indices. The bug caused offsets to be generated without the last element,
    resulting in an incorrect output shape (missing the last bag).
    """
    # Setup parameters
    num_embeddings = 10
    embedding_dim = 3
    mode = 'sum'
    include_last_offset = True
    
    # Initialize EmbeddingBag
    embedding_sum = nn.EmbeddingBag(
        num_embeddings, 
        embedding_dim, 
        mode=mode, 
        include_last_offset=include_last_offset
    )
    
    # Create 2D input: 2 bags (rows), 4 indices each
    # Total indices size = 8
    input_indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    
    # Execute forward pass
    output = embedding_sum(input_indices)
    
    # Verification
    # Expected behavior:
    # Offsets should be [0, 4, 8] (length 3).
    # Number of bags = len(offsets) - 1 = 2.
    # Therefore, output shape should be (2, 3).
    
    # Buggy behavior:
    # Offsets were [0, 4] (length 2).
    # Number of bags = len(offsets) - 1 = 1.
    # Output shape would be (1, 3).
    
    expected_shape = (2, embedding_dim)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {output.shape}. "
        "This suggests that offsets were not generated correctly for 2D input "
        "when include_last_offset is True."
    )

if __name__ == "__main__":
    test_embeddingbag_2d_input_with_include_last_offset()
    print("Test passed.")