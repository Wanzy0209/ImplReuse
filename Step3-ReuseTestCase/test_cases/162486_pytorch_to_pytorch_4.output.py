import torch
from torch import set_default_device

def test_unbind_with_default_device(device: str = 'cuda'):
    set_default_device(device) # Set the default device context, similar to the bug report

    # Create a tensor. Due to set_default_device, this tensor will be on 'device'.
    x = torch.randn(100, 3)

    # Call torch.unbind (the similar API)
    # This replaces the random_split call from the original bug report
    try:
        unbound_tensors = torch.unbind(x, dim=0)
        
        # Assertions to verify correctness
        assert len(unbound_tensors) == 100, "Unbound length mismatch"
        assert unbound_tensors[0].device.type == device, f"Device mismatch: expected {device}, got {unbound_tensors[0].device.type}"
        
        print(f"Device {device} worked for torch.unbind.")
    except Exception as e:
        print(f"Device {device} failed for torch.unbind: {e}")

# Run tests
test_unbind_with_default_device(device='cpu')

if torch.cuda.is_available():
    test_unbind_with_default_device(device='cuda')
else:
    print("CUDA not available, skipping CUDA test.")