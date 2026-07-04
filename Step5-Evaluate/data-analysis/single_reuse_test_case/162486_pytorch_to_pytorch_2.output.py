import torch
import torch.nn as nn

def test_adaptive_max_pool(device: str = 'cuda'):
    # set_default_device is not available in older PyTorch versions.
    # We manually move tensors to the device instead.

    # Create input tensor and move to device
    # Shape: (Batch Size, Channels, Length)
    input_tensor = torch.randn(1, 64, 8).to(device)

    # Initialize the AdaptiveMaxPool1d layer and move to device
    layer = nn.AdaptiveMaxPool1d(output_size=5).to(device)

    # Perform forward pass
    output = layer(input_tensor)

    # Verify output shape
    # Input length is 8, target output size is 5
    assert output.shape == (1, 64, 5), f"Expected shape (1, 64, 5), got {output.shape}"

    print(f"Device {device} worked.")

# Test with CPU
test_adaptive_max_pool(device='cpu')

# Test with CUDA if available
if torch.cuda.is_available():
    test_adaptive_max_pool(device='cuda')
else:
    print("CUDA not available, skipping CUDA test.")