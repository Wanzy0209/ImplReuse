import torch

def test_cumsum(device):
    # Adaptation: Test with a 0-dimensional (scalar) tensor input,
    # similar to how the original bug tested a scalar index tensor.
    x = torch.tensor(5, device=device)
    try:
        # For 0-dim tensors, dim must be 0 or -1
        output = torch.cumsum(x, dim=0)
        print(f"cumsum test succeeds for device: {device}. output: {output}, shape: {output.shape}")
    except Exception as e:
        print(f"cumsum test fails for device: {device}: {e}")

test_cumsum(device="cpu")
test_cumsum(device="mps")