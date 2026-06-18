import warnings
warnings.simplefilter('error')
import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")

# Setup input with requires_grad=True, similar to the issue's 'a' tensor
# Shape (Batch, Channels, Height, Width) = (2, 3, 4, 4)
# This ensures size > 1 to avoid the ValueError in the provided SyncBatchNorm snippet
x = torch.rand((2, 3, 4, 4))
x.requires_grad_()

# Instantiate the Similar API: torch.nn.SyncBatchNorm
layer = nn.SyncBatchNorm(3)

# Define a closure similar to the LBFGS example in the issue
def closure():
    # Zero gradients (simulating optimizer.zero_grad)
    if layer.weight is not None:
        layer.weight.grad = None
    if layer.bias is not None:
        layer.bias.grad = None
    
    # Forward pass through the Similar API
    # The snippet logic (contiguity check, size check) runs here internally
    output = layer(x)
    
    # Compute loss
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    return loss

# The issue was triggered by LBFGS calling float(closure()).
# We test if using SyncBatchNorm in this context triggers the warning.
try:
    # This mimics the internal behavior of LBFGS that caused the bug
    loss_val = closure()
    
    # Explicitly test the scalar conversion mentioned in the bug report
    # to ensure it doesn't raise the warning with this API
    scalar_loss = float(loss_val)
    
    print("Test passed: No warnings raised when converting loss to scalar.")
    print(f"Scalar loss: {scalar_loss}")
    
except UserWarning as e:
    print(f"Test failed: UserWarning raised - {e}")
except Exception as e:
    print(f"Test failed: Exception raised - {e}")