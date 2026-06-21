import torch
import torch.nn as nn

# Adapted test case for torch.nn.CosineEmbeddingLoss
# Based on the original bug report regarding 4D input handling in F.pad

def test_cosine_embedding_loss_4d():
    # Create 4D inputs similar to the dimensionality in the original bug report
    input1 = torch.randn(2, 2, 2, 2)
    input2 = torch.randn(2, 2, 2, 2)
    # FIX: CosineEmbeddingLoss expects a 1D target tensor containing values 1 or -1.
    # The batch size is 2 (the first dimension of the inputs).
    target = torch.ones(2)

    criterion = nn.CosineEmbeddingLoss()

    try:
        # Attempt to compute loss with 4D inputs
        output = criterion(input1, input2, target)
        
        # Verify the output is a scalar (default reduction='mean')
        assert output.dim() == 0, "Output should be a scalar with default reduction"
        print("Test passed: CosineEmbeddingLoss handles 4D inputs correctly.")
        
    except NotImplementedError as e:
        # Check if the error message is misleading (similar to the original bug)
        error_msg = str(e)
        print(f"NotImplementedError raised: {error_msg}")
        # In the original bug, the error message claimed support for 4D/5D but failed.
        # Here we verify if the error message accurately reflects the API's capabilities.
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_cosine_embedding_loss_4d()