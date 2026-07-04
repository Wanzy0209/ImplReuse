import torch
import numpy as np

# Fix: torch.set_default_device is not available in older PyTorch versions.
# We explicitly define the device and pass it to tensor creation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a symmetric matrix for symeig
n = 1024
A = torch.randn(n, n, dtype=torch.float, device=device)
A = (A + A.T) / 2  # Ensure symmetry

# Call torch.symeig
# Returns (eigenvalues, eigenvectors)
# Fix: Added eigenvectors=True to ensure both e and v are returned for unpacking
e, v = torch.symeig(A, eigenvectors=True)
print("torch.symeig Eigenvalues:", e.shape, e.stride())
print("torch.symeig Eigenvectors:", v.shape, v.stride())

# Call numpy.linalg.eigh for comparison
e_np, v_np = np.linalg.eigh(A.cpu().numpy())
print("numpy.linalg.eigh Eigenvalues:", e_np.shape, e_np.strides)
print("numpy.linalg.eigh Eigenvectors:", v_np.shape, v_np.strides)