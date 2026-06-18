import torch

def test_cummin(device):
    # Adaptation: The original bug involved a scalar (0-dimensional) tensor passed as an index.
    # We test torch.cummin with a scalar input tensor to verify similar handling of 0-dimensional tensors.
    x = torch.tensor(1.0, device=device)  # zero-dimensional input tensor
    try:
        # For a 0-dimensional tensor, dim must be 0 or -1
        values, indices = torch.cummin(x, dim=0)
        print(f"cummin test succeeds for device: {device}. values: {values}, indices: {indices}")
    except Exception as e:
        print(f"cummin test fails for device: {device}: {e}")

test_cummin(device="cpu")
test_cummin(device="mps")