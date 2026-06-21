import torch

def test_arange(device):
    # Check if the requested device is MPS and if it is available
    if device == "mps":
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print(f"Skipping test for device: {device} (not available in this environment)")
            return

    # Adaptation: Create a zero-dimensional tensor on the specified device
    # and pass it to torch.arange to verify consistent behavior across devices.
    x = torch.tensor(3.0, device=device)
    try:
        output = torch.arange(x)
        print(f"arange test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"arange test fails for device: {device}: {e}")

test_arange(device="cpu")
test_arange(device="mps")