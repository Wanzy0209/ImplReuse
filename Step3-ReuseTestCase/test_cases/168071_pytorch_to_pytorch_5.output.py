import torch
import torch.nn as nn

# Adapted test case for torch.nn.CosineEmbeddingLoss based on Issue 168071
# The original issue involved a crash when padding a tensor with a 0-shape dimension (e.g., (6, 0)).
# This test verifies if CosineEmbeddingLoss handles inputs with a 0-shape dimension correctly
# without crashing due to dimension arithmetic errors.

def test_cosine_embedding_loss_zero_dim():
    # Create inputs with a 0-shape dimension, similar to the original bug report's input (6, 0)
    input1 = torch.zeros((6, 0))
    input2 = torch.zeros((6, 0))
    target = torch.ones(6)  # Target labels

    loss_fn = nn.CosineEmbeddingLoss()

    try:
        # Attempt to compute loss
        loss = loss_fn(input1, input2, target)
        print(f"Test passed. Loss computed: {loss}")
        # We expect the loss to be computed (likely resulting in NaN or 0 depending on implementation)
        # rather than crashing with a "negative output size" error.
        assert True 
    except RuntimeError as e:
        print(f"Test failed with RuntimeError: {e}")
        assert False, f"CosineEmbeddingLoss crashed on 0-shape input: {e}"

if __name__ == "__main__":
    test_cosine_embedding_loss_zero_dim()