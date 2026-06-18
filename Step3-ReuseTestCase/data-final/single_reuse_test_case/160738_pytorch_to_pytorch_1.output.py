import torch

def test_std(device):
    x = torch.tensor(3.0, device=device)
    try:
        output = torch.std(x, dim=0)
        print(f"std test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"std test fails for device: {device}: {e}")

test_std(device = "cpu")
test_std(device = "mps")