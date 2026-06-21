import torch
import sys

# Handle missing torch._inductor module (common in older PyTorch versions or specific builds)
try:
    import torch._inductor.config as inductor_config
except ImportError:
    print("Skipping test: torch._inductor module not found. This test requires a PyTorch version with Inductor support (typically PyTorch 2.0+).")
    sys.exit(0)

# Setup dimensions
# Using a square matrix for lobpcg. 
# N=1024 is chosen to be large enough to trigger Triton kernels but small enough for a quick test.
N = 1024
k_eig = 10

# Create a symmetric positive definite matrix required by lobpcg
# A = X @ X.T + epsilon * I
torch.manual_seed(42)

# Ensure CUDA is available before attempting to use it
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available.")
    sys.exit(0)

X = torch.randn(N, N, dtype=torch.float32).cuda()
A = X @ X.T + 1e-3 * torch.eye(N, device='cuda')

# Define the function to compile using torch.lobpcg
def lobpcg_func(A):
    # lobpcg returns a tuple of (eigenvalues, eigenvectors)
    return torch.lobpcg(A, k=k_eig)

# Apply the specific configuration from the bug report that triggered the failure
with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_fallback_to_aten=False,
):
    compiled = torch.compile(lobpcg_func, dynamic=False)
    
    # Run the compiled function
    eigenvalues, eigenvectors = compiled(A)

# Assertions to verify the output
assert eigenvalues.shape == (k_eig,), f"Expected eigenvalues shape ({k_eig},), got {eigenvalues.shape}"
assert eigenvectors.shape == (N, k_eig), f"Expected eigenvectors shape ({N}, {k_eig}), got {eigenvectors.shape}"
assert torch.is_tensor(eigenvalues), "Eigenvalues should be a tensor"
assert torch.is_tensor(eigenvectors), "Eigenvectors should be a tensor"

print("Test passed.")