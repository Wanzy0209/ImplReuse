import torch

def test_std(device):
    # Check for MPS availability to avoid RuntimeError on unsupported systems
    if device == "mps":
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            print(f"std test skipped for device: {device} (MPS not available on this system)")
            return

    try:
        # Move tensor creation inside try block to handle potential device errors gracefully
        x = torch.tensor(3.0, device=device)
        output = torch.std(x, dim=0)
        print(f"std test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"std test fails for device: {device}: {e}")

test_std(device = "cpu")
test_std(device = "mps")