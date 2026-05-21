import torch
from torch import set_default_device
from torch.nn.utils.rnn import pack_sequence


def test_pack_sequence_default_device(device: str = 'cuda'):
    set_default_device(device) # Set the default device before creating tensors

    # Create sequences of varying lengths.
    # Since set_default_device is called, these tensors will be created on 'device'.
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5])
    c = torch.tensor([6])

    # Call the API under test
    # This might fail if the API does not handle the default device correctly
    # (e.g., if it creates internal tensors on CPU while inputs are on CUDA, or vice versa)
    packed = pack_sequence([a, b, c])

    # Verify the output
    assert isinstance(packed, torch.nn.utils.rnn.PackedSequence)
    assert packed.data.device.type == device
    
    print(f"Device {device} worked.")

# Test with CPU first
test_pack_sequence_default_device(device='cpu')

# Test with CUDA to verify if the similar API has the same issue
test_pack_sequence_default_device(device='cuda')