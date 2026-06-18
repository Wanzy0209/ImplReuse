import torch

# Configure Dynamo settings from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(13653)

# Setup inputs for torch.lobpcg
# A must be symmetric positive definite
A = torch.randn(5, 5, dtype=torch.float32)
A = A @ A.T + torch.eye(5)
# X is the initial approximation to the eigenvectors
X = torch.randn(5, 2, dtype=torch.float32)

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

def test_lobpcg(A, X, sentinel):
    # Call the similar API: torch.lobpcg
    # We pass scalar arguments (niter, tol) to exercise scalar handling paths
    eigenvalues, eigenvectors = torch.lobpcg(A, k=2, X=X, niter=5, tol=1e-5)
    
    # Perform scalar operations similar to the original fuzzed_program
    # to ensure scalar capture logic is exercised
    scalar_val = eigenvalues[0].item()
    result = scalar_val * sentinel
    
    if result.is_complex():
        result = result.real
        
    return eigenvalues, eigenvectors, result

args = (A, X, sentinel)

# Run eager mode
result_original = test_lobpcg(*args)
print(' eager success')

# Run compiled mode
compiled_program = torch.compile(test_lobpcg, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')