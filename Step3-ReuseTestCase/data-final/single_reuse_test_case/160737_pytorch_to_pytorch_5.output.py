import torch

def test_cummax(device):
    # Adaptation: Testing cummax with a zero-dimensional (scalar) input tensor.
    # This mirrors the original bug where index_select failed with a scalar index tensor.
    x = torch.tensor(1.0, device=device)  # zero-dimensional input tensor
    try:
        # For a scalar tensor, dim must be 0 or -1
        output, indices = torch.cummax(x, dim=0)
        print(f"cummax test succeeds for device: {device}. output shape: {output.shape}")
    except Exception as e:
        print(f"cummax test fails for device: {device}: {e}")

test_cummax(device="cpu")
test_cummax(device="mps")