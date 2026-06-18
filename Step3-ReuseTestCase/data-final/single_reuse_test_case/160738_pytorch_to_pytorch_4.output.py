import torch

def test_svd(device):
    # Create a zero-dimensional tensor similar to the original bug report
    x = torch.tensor(3.0, device=device)
    try:
        # torch.svd does not take a 'dim' argument, but we test if it handles
        # zero-dimensional input consistently between CPU and MPS
        output = torch.svd(x)
        print(f"svd test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"svd test fails for device: {device}: {e}")

test_svd(device="cpu")
test_svd(device="mps")