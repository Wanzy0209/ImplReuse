import torch
import torch.nn as nn

def test_cosine_embedding_loss_mps():
    """
    Test case to verify torch.nn.CosineEmbeddingLoss functionality on the MPS device.
    This test is derived from a bug report (Issue 160237) where aten::grid_sampler_3d 
    was not implemented for MPS. This test checks if the similar API, 
    CosineEmbeddingLoss, is implemented and functional on MPS.
    """
    # Check if MPS is available to run the test
    if not torch.backends.mps.is_available():
        print("MPS device is not available. Skipping test.")
        return

    device = torch.device("mps")

    # Create dummy input tensors
    # CosineEmbeddingLoss expects input tensors x1, x2 with the same size (N, D) or (D,)
    batch_size = 4
    embedding_dim = 128
    
    input1 = torch.randn(batch_size, embedding_dim, device=device)
    input2 = torch.randn(batch_size, embedding_dim, device=device)
    
    # Target tensor must contain values 1 or -1
    target = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)

    # Initialize the loss function
    loss_fn = nn.CosineEmbeddingLoss()

    # Attempt to compute the loss on MPS
    try:
        loss = loss_fn(input1, input2, target)
        
        # Verify the output is a scalar tensor (0-dim)
        assert loss.dim() == 0, f"Expected scalar output, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss resulted in NaN"
        
        print(f"Test Passed. CosineEmbeddingLoss on MPS is functional. Loss value: {loss.item()}")
        
    except NotImplementedError as e:
        print(f"Test Failed. NotImplementedError caught: {e}")
        raise

if __name__ == "__main__":
    test_cosine_embedding_loss_mps()