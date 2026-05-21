import torch

def get_ndim(x):
    """
    Helper function leveraging the logic of the similar API 'tf.keras.backend.ndim'.
    Returns the rank of the tensor.
    """
    return x.ndim

def test_embeddingbag_2d_include_last_offset():
    """
    Test case for Issue 167974.
    Verifies that EmbeddingBag with include_last_offset=True generates correct
    offsets for 2D input, ensuring the output shape matches the number of rows.
    """
    # Setup parameters
    num_embeddings = 10
    embedding_dim = 3
    mode = 'sum'
    include_last_offset = True

    # Initialize EmbeddingBag
    embedding_sum = torch.nn.EmbeddingBag(
        num_embeddings, 
        embedding_dim, 
        mode=mode, 
        include_last_offset=include_last_offset
    )

    # Create 2D input tensor (similar to the structure in the bug report and similar API examples)
    input_indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)

    # Verify input rank using the helper derived from the similar API
    # The bug specifically occurs when the input is 2D
    assert get_ndim(input_indices) == 2, "Test input must be 2D to reproduce the issue"

    # Run the forward pass
    # Bug: Offsets are generated as [0, 4] instead of [0, 4, 8]
    # Expected: Offsets should be [0, 4, 8] to cover all 8 elements in 2 bags
    output = embedding_sum(input_indices)

    # Assertions
    # If offsets are [0, 4] (buggy), the kernel sees 1 bag (offsets size - 1).
    # If offsets are [0, 4, 8] (correct), the kernel sees 2 bags.
    # We expect the output to have 2 rows (one for each bag).
    expected_shape = (2, embedding_dim)
    assert output.shape == expected_shape, (
        f"Shape mismatch. Expected {expected_shape} (2 bags), got {output.shape}. "
        "This indicates offsets were not generated correctly for include_last_offset=True."
    )

    # Verify values to ensure correct summation logic
    # Bag 0: indices [1, 2, 4, 5]
    # Bag 1: indices [4, 3, 2, 9]
    weights = embedding_sum.weight
    expected_bag_0 = weights[1] + weights[2] + weights[4] + weights[5]
    expected_bag_1 = weights[4] + weights[3] + weights[2] + weights[9]
    
    expected_output = torch.stack([expected_bag_0, expected_bag_1])
    
    assert torch.allclose(output, expected_output), (
        "Output values incorrect. The offsets likely did not cover the full input range."
    )

if __name__ == "__main__":
    test_embeddingbag_2d_include_last_offset()
    print("Test passed.")