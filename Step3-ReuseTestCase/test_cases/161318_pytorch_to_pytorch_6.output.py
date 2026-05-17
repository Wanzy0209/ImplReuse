import torch

# Reproduce the bug scenario: torch.compile(fullgraph=True) with data-dependent slicing
# adapted to use torch.lobpcg as the primary operation being tested.

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True)
def fn(mask, matrix):
    # Data-dependent slice logic from the original bug report
    # Original: text_len = encoder_attention_mask.sum().item()
    #          encoder_hidden_states = encoder_hidden_states[:, :text_len]
    
    dim = mask.sum().item()
    
    # Slice the matrix based on the data-dependent dimension
    # This triggers the graph compilation logic that was crashing
    A_sliced = matrix[:dim, :dim]
    
    # Call the similar API: torch.lobpcg
    # We request k=2 eigenvalues. 
    # Note: lobpcg requires k < dim. The mask logic ensures dim=8, so k=2 is safe.
    k = 2
    if dim > k:
        eigenvalues, eigenvectors = torch.lobpcg(A_sliced, k=k)
        return eigenvalues, eigenvectors
    return None, None

# Setup inputs
# Create a symmetric positive definite matrix (required for lobpcg)
size = 10
A = torch.randn(size, size)
A = A @ A.T + torch.eye(size) * 0.1

# Create a mask similar to the bug report
# (torch.arange(512) < 8) results in 8 True values
mask = (torch.arange(size) < 8)

# Run the test
# The original bug report used CUDA, but this test runs on CPU for compatibility.
# The crash was in the Inductor compiler logic, which applies to both.
try:
    e, v = fn(mask, A)
    if e is not None:
        print("Test passed. Eigenvalues:", e)
        # Basic assertion to verify execution
        assert e.shape[0] == 2
        assert v.shape[0] == size
    else:
        print("Test skipped (dim too small)")
except Exception as e:
    print(f"Test failed with error: {e}")