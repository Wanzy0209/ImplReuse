import torch

def test_bug(device: str = 'cuda'):
    # set_default_device is not available in older PyTorch versions.
    # We explicitly pass the device to the tensor constructor instead.

    # Create a symmetric matrix (required for symeig)
    A = torch.randn(10, 10, device=device)
    A = A @ A.T # Ensure symmetry

    # Call the similar API
    eigenvalues, eigenvectors = torch.symeig(A, eigenvectors=True) # BUG CHECK

    print(f"Device {device} worked.")

test_bug(device='cpu') # works
test_bug(device='cuda') # check for error