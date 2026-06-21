import torch
from torch.nn.utils.rnn import pack_sequence


def test_pack_sequence_default_device(device: str = 'cuda'):
    # set_default_device is not available in older PyTorch versions.
    # We explicitly create tensors on the target device instead to ensure compatibility.

    # Create sequences of varying lengths on the specified device.
    a = torch.tensor([1, 2, 3], device=device)
    b = torch.tensor([4, 5], device=device)
    c = torch.tensor([6], device=device)

    # Call the API under test
    packed = pack_sequence([a, b, c])

    # Verify the output
    assert isinstance(packed, torch.nn.utils.rnn.PackedSequence)
    assert packed.data.device.type == device
    
    print(f"Device {device} worked.")

# Test with CPU first
test_pack_sequence_default_device(device='cpu')

# Test with CUDA to verify if the similar API has the same issue
if torch.cuda.is_available():
    test_pack_sequence_default_device(device='cuda')
else:
    print("CUDA not available, skipping CUDA test.")