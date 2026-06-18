import torch

def test_logcumsumexp(device):
    # Adapted to test 0-dimensional input tensor (scalar tensor)
    # based on the extracted logic handling len(x.get_size()) == 0
    x = torch.tensor(1.0, device=device)
    try:
        # For 0-dim tensors, dim must be 0 or -1
        output = torch.logcumsumexp(x, dim=0)
        print(f"logcumsumexp test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"logcumsumexp test fails for device: {device}: {e}")

test_logcumsumexp(device="cpu")
test_logcumsumexp(device="mps")