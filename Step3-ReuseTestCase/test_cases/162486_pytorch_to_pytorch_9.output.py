import torch
import torch.nn.functional as F
from torch import set_default_device

def test_rrelu(device: str = 'cuda'):
    set_default_device(device)

    # Create input tensor. If device is 'cuda', this is a CUDA tensor.
    input_tensor = torch.randn(100, 3)

    # Call rrelu. training=True ensures randomness is used.
    # The bug in random_split was related to internal tensor creation on the wrong device.
    # We want to see if rrelu handles the default device setting correctly.
    output = F.rrelu(input_tensor, training=True)

    # Basic assertion to ensure it ran
    assert output.shape == input_tensor.shape
    print(f"Device {device} worked.")

test_rrelu(device='cpu')
test_rrelu(device='cuda')