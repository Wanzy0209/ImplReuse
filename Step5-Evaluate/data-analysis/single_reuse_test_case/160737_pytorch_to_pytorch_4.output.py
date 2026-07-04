import torch

def test_cummin(device):
    # Adaptation: The original bug involved a scalar (0-dimensional) tensor passed as an index.
    # We test torch.cummin with a scalar input tensor to verify similar handling of 0-dimensional tensors.
    try:
        x = torch.tensor(1.0, device=device)  # zero-dimensional input tensor
        # For a 0-dimensional tensor, dim must be 0 or -1
        values, indices = torch.cummin(x, dim=0)
        print(f"cummin test succeeds for device: {device}. values: {values}, indices: {indices}")
    except Exception as e:
        print(f"cummin test fails for device: {device}: {e}")

test_cummin(device="cpu")

# Check for MPS availability before running the test to avoid RuntimeError
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    test_cummin(device="mps")
else:
    print("Skipping MPS test: MPS device not available.")