import torch

def test_minimum(device):
    # Adaptation: torch.minimum is an element-wise operation requiring two inputs.
    # We test it with zero-dimensional tensors to check for similar backend handling issues.
    x = torch.tensor(3.0, device=device)
    y = torch.tensor(2.0, device=device)
    try:
        output = torch.minimum(x, y)
        print(f"minimum test succeeds for device: {device}. output: {output}")
    except Exception as e:
        print(f"minimum test fails for device: {device}: {e}")

test_minimum(device="cpu")

# Check for MPS availability before running the test to avoid RuntimeError
if torch.backends.mps.is_available():
    test_minimum(device="mps")
else:
    print("Skipping MPS test: MPS device not available or PyTorch not built with MPS support.")