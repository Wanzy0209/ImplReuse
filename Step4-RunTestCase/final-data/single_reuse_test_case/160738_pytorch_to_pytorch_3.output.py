import torch

def test_dist(device):
    # Adaptation: torch.dist does not take a 'dim' argument like torch.var.
    # We test the behavior with zero-dimensional tensors, which was the root cause
    # of the issue in torch.var (handling of 0-d tensors).
    x = torch.tensor(3.0, device=device)
    y = torch.tensor(4.0, device=device)
    try:
        output = torch.dist(x, y)
        print(f"dist test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"dist test fails for device: {device}: {e}")

test_dist(device="cpu")
if torch.backends.mps.is_available():
    test_dist(device="mps")