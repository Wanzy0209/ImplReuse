import torch
import torch.nn as nn

# The original bug occurs when a batch dimension is larger than uint16 max (2**16)
# We test if torch.nn.CosineEmbeddingLoss handles this scenario correctly on CUDA.

if torch.cuda.is_available():
    # Setup inputs with a batch dimension of 2**16
    batch_size = 2**16
    dim = 2
    
    input1 = torch.rand(batch_size, dim, device="cuda")
    input2 = torch.rand(batch_size, dim, device="cuda")
    # Target must be 1 or -1 for CosineEmbeddingLoss
    target = torch.randint(0, 2, (batch_size,), device="cuda") * 2 - 1

    loss_fn = nn.CosineEmbeddingLoss()

    try:
        # Attempt to compute loss
        loss = loss_fn(input1, input2, target)
        print(f"CosineEmbeddingLoss test passed with batch size {batch_size}. Loss: {loss.item()}")
    except RuntimeError as e:
        print(f"CosineEmbeddingLoss test failed with batch size {batch_size}: {e}")
else:
    print("CUDA not available, skipping test.")