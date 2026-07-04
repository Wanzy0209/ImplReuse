import torch

def test_norm(device):
    x = torch.tensor(3.0, device=device)
    try:
        output = torch.norm(x, dim=0)
        print(f"norm test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"norm test fails for device: {device}: {e}")

test_norm(device="cpu")

if torch.backends.mps.is_available():
    test_norm(device="mps")
else:
    print("norm test skipped for device: mps (not available)")