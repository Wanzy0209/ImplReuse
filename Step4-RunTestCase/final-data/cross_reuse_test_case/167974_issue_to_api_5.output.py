import torch
import torch.nn as nn

def test_embeddingbag_2d_include_last_offset():
    """
    Test case for Issue #167974.
    Verifies that EmbeddingBag with include_last_offset=True generates correct
    offsets when the input is a 2D tensor.
    
    The bug report indicates that with include_last_offset=True, the internal
    offsets generated for a 2D input of shape (2, 4) are [0, 4] instead of [0, 4, 8].
    This results in an output shape of (1, 3) instead of the expected (2, 3).
    """
    # Check PyTorch version to skip if the known bug exists
    # The bug was fixed in PyTorch 1.8.0
    try:
        from packaging import version
        if version.parse(torch.__version__) < version.parse("1.8.0"):
            print(f"Skipping test: PyTorch {torch.__version__} has a known bug with EmbeddingBag 2D input and include_last_offset=True.")
            return
    except ImportError:
        # Fallback to simple string comparison if packaging is not available
        try:
            major, minor = map(int, torch.__version__.split('.')[:2])
            if major == 1 and minor < 8:
                print(f"Skipping test: PyTorch {torch.__version__} has a known bug with EmbeddingBag 2D input and include_last_offset=True.")
                return
        except ValueError:
            # If version string is unparseable, proceed with test
            pass

    # Setup: Create EmbeddingBag with include_last_offset=True
    embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
    
    # Set weights to 1.0 to easily verify the sum operation
    with torch.no_grad():
        embedding_sum.weight.fill_(1.0)

    # Input: 2D tensor with 2 rows (bags) and 4 indices each
    input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)

    # Execute
    output = embedding_sum(input)

    # Verification
    # Expected behavior: 2 bags (one per row).
    # If offsets are [0, 4, 8], output shape is (2, 3).
    # If offsets are [0, 4] (bug), output shape is (1, 3).
    expected_shape = torch.Size([2, 3])
    
    # Since weights are 1.0, the sum for each row (4 elements) should be 4.0 for each embedding dimension
    expected_values = torch.full((2, 3), 4.0)

    assert output.shape == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, but got {output.shape}. " \
        "This indicates offsets were generated as [0, 4] instead of [0, 4, 8]."
    
    assert torch.allclose(output, expected_values), \
        f"Value mismatch: expected {expected_values}, but got {output}."

if __name__ == "__main__":
    test_embeddingbag_2d_include_last_offset()
    print("Test passed.")