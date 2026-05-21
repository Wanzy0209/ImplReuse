import torch
from torch.nn.functional import dropout3d
from torch import set_default_device

def test_dropout_default_device(device: str = 'cuda'):
    set_default_device(device)

    # Create a 5D tensor (Batch, Channel, Depth, Height, Width)
    # Since default device is set, this tensor will be created on 'device'
    x = torch.randn(2, 3, 4, 5, 6)

    # Call dropout3d
    # This operation involves random number generation to create the mask.
    # We verify if it handles the default device context correctly.
    try:
        output = dropout3d(x, p=0.5, training=True)

        # Basic sanity checks
        assert output.shape == x.shape, "Shape mismatch after dropout"
        assert output.device == x.device, f"Device mismatch: output on {output.device}, expected {x.device}"
        
        print(f"Device {device} worked.")
    except Exception as e:
        print(f"Device {device} failed with error: {e}")
        raise

# Run tests
if torch.cuda.is_available():
    test_dropout_default_device(device='cpu')
    test_dropout_default_device(device='cuda')
else:
    print("CUDA not available, skipping CUDA test.")
    test_dropout_default_device(device='cpu')