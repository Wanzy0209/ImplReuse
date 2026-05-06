import torch

def test_clamp(device):
    x = torch.randn(2, 3, device=device)
    min_t = torch.randn(1, 2, 3, dtype=x.dtype, device=device)
    max_t = torch.randn(1, 2, 3, dtype=x.dtype, device=device)
    try:
        torch.clamp(x, min=min_t, max=max_t)
        print(f"clamp test succeeds for device: {device}")
    except RuntimeError as e:
        print(f"clamp test fails for device: {device}: {e}")

test_clamp(device = "cpu")
test_clamp(device = "mps")