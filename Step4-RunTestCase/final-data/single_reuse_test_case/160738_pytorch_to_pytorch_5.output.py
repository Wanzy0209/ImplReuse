import torch

def test_maximum(device):
    # Check if MPS is available if the requested device is mps
    if device == "mps" and not torch.backends.mps.is_available():
        print(f"maximum test skipped for device: {device} (MPS not supported in this environment)")
        return

    # Adapted from the original torch.var test case.
    # torch.maximum is an element-wise operation and does not accept a 'dim' argument.
    # We test if it handles zero-dimensional tensors correctly on different devices.
    x = torch.tensor(3.0, device=device)
    y = torch.tensor(5.0, device=device)
    try:
        output = torch.maximum(x, y)
        print(f"maximum test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"maximum test fails for device: {device}: {e}")

test_maximum(device="cpu")
test_maximum(device="mps")