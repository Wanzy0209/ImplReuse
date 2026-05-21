import torch
from torch import set_default_device

def test_bug(device: str = 'cuda'):
    set_default_device(device) # if device is 'cuda' then 'symeig' might throw an error

    # Create a symmetric matrix (required for symeig)
    # torch.randn respects the default device set above
    A = torch.randn(10, 10)
    A = A @ A.T # Ensure symmetry

    # Call the similar API
    eigenvalues, eigenvectors = torch.symeig(A, eigenvectors=True) # BUG CHECK

    print(f"Device {device} worked.")

test_bug(device='cpu') # works
test_bug(device='cuda') # check for error