import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

def test_clip_grad_value_default_device(device: str = 'cuda'):
    # Convert string to torch.device object for compatibility
    device_obj = torch.device(device)

    # Create a simple linear layer and explicitly move to device
    # This replaces the functionality of set_default_device
    layer = nn.Linear(10, 2).to(device_obj)

    # Create input and target and explicitly move to device
    x = torch.randn(5, 10).to(device_obj)
    y = torch.randn(5, 2).to(device_obj)

    # Forward and backward pass to populate gradients
    output = layer(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()

    # Ensure gradients are populated
    assert layer.weight.grad is not None

    # Call the API under test
    # This is the adaptation of the original bug report's call site
    clip_grad_value_(layer.parameters(), clip_value=0.5)

    # Verify the clipping worked
    # The maximum absolute value in the gradients should be <= 0.5
    max_grad = layer.weight.grad.abs().max().item()
    assert max_grad <= 0.5, f"Gradients not clipped correctly on {device}. Max grad: {max_grad}"

    print(f"Device {device} worked.")

# Run tests
test_clip_grad_value_default_device('cpu')

if torch.cuda.is_available():
    test_clip_grad_value_default_device('cuda')
else:
    print("CUDA not available, skipping CUDA test.")