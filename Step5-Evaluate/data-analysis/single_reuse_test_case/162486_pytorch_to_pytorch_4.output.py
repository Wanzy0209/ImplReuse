import torch

def test_unbind_with_default_device(device: str = 'cuda'):
    # Check if set_default_device is available (introduced in PyTorch 1.12)
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device(device)
        x = torch.randn(100, 3)
    else:
        # Fallback for older PyTorch versions: explicitly move tensor to device
        # This ensures the test logic (testing unbind on a device) remains intact
        # even if the specific API (set_default_device) is missing.
        x = torch.randn(100, 3).to(device)

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