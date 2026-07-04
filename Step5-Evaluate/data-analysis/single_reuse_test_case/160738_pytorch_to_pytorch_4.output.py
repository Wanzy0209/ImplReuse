import torch

def test_svd(device):
    # Create a zero-dimensional tensor similar to the original bug report
    try:
        # Move tensor creation inside try block to catch device errors
        x = torch.tensor(3.0, device=device)
        # torch.svd does not take a 'dim' argument, but we test if it handles
        # zero-dimensional input consistently between CPU and MPS
        output = torch.svd(x)
        print(f"svd test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"svd test fails for device: {device}: {e}")

test_svd(device="cpu")

# Check for MPS availability to prevent runtime error on unsupported environments
if torch.backends.mps.is_available():
    test_svd(device="mps")
else:
    print("Skipping MPS test: MPS device not available.")