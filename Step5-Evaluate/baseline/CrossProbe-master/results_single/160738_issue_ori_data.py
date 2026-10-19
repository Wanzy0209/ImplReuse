# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

def test_var(device):
    x = torch.tensor(3.0, device=device)
    try:
        output = torch.var(x, dim=0)
        print(f"var test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"var test fails for device: {device}: {e}")

test_var(device = "cpu")
test_var(device = "mps")