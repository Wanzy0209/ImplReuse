import torch

def test_norm(device):
    x = torch.tensor(3.0, device=device)
    try:
        output = torch.norm(x, dim=0)
        print(f"norm test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"norm test fails for device: {device}: {e}")

test_norm(device="cpu")
test_norm(device="mps")