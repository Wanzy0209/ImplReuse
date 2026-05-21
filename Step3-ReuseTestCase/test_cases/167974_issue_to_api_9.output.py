import torch

def test_embeddingbag_2d_include_last_offset():
    """
    Test case for Issue 167974:
    EmbeddingBag sum has incorrect offsets when input is 2D and include_last_offset is set to True.
    
    When include_last_offset is True, the offsets should have one additional element
    representing the size of the indices. For a 2D input, the framework auto-generates
    offsets. This test verifies that these auto-generated offsets adhere to the
    include_last_offset flag.
    """
    # Parameters
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

    # 2D Input: 2 rows, 4 columns
    # Flattened indices: [1, 2, 4, 5, 4, 3, 2, 9]
    input_2d = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)

    # Calculate expected output using 1D input with explicit correct offsets.
    # With include_last_offset=True, offsets should be [0, 4, 8].
    input_1d = input_2d.view(-1)
    correct_offsets = torch.tensor([0, 4, 8], dtype=torch.long)
    expected_output = embedding_sum(input_1d, offsets=correct_offsets)

    # Calculate actual output using 2D input (which triggers auto-generation of offsets).
    # If the bug is present, offsets are generated as [0, 4], causing the second bag
    # to be ignored or processed incorrectly.
    actual_output = embedding_sum(input_2d)

    # Assert that the outputs match.
    # If the bug exists, actual_output will likely have a different shape (1, 3) vs (2, 3)
    # or incorrect values.
    assert torch.equal(actual_output, expected_output), \
        f"EmbeddingBag 2D input with include_last_offset=True failed.\nExpected shape: {expected_output.shape}, Actual shape: {actual_output.shape}\nExpected: {expected_output}\nActual: {actual_output}"

if __name__ == "__main__":
    test_embeddingbag_2d_include_last_offset()
    print("Test passed.")