import torch

# Adapted test case for torch.nn.Embedding
# The original bug report concerned torch.nn.EmbeddingBag with include_last_offset=True.
# torch.nn.Embedding is a similar API but does not support 'mode' or 'include_last_offset'.
# This test verifies that torch.nn.Embedding correctly handles the 2D input tensor
# used in the original reproduction script.

def test_embedding_2d_input():
    num_embeddings = 10
    embedding_dim = 3
    
    # Initialize Embedding layer (similar parameters to EmbeddingBag)
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    
    # Input tensor from the original bug report
    input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    
    # Perform the lookup
    output = embedding(input)
    
    # Verify output shape
    # Input shape is (2, 4), so output shape should be (2, 4, 3)
    expected_shape = (2, 4, embedding_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    
    print("Test passed: torch.nn.Embedding handles 2D input correctly.")

if __name__ == "__main__":
    test_embedding_2d_input()