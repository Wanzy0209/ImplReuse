import torch

# Handle missing torch.compile for older PyTorch versions (< 2.0)
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found. Running in eager mode (mocking torch.compile).")
    # Mock torch.compile to return the function as-is, allowing the test to proceed
    torch.compile = lambda func, **kwargs: func

class Config:
    def __repr__(self):
        return "Config()"

def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    r = repr(config)
    
    # Use the similar API torch.lobpcg inside the compiled function
    # Create a symmetric positive definite matrix A
    A = x @ x.T + torch.eye(x.size(0))
    # Compute the largest eigenvalue
    eigenvalues, _ = torch.lobpcg(A, k=1)
    
    # Combine the results to ensure both operations are traced
    return eigenvalues.sum() * len(r)

config = Config()
x = torch.randn(2, 2)

# Compile the function with fullgraph=True
# If torch.compile was mocked, this simply returns the forward function
compiled = torch.compile(forward, fullgraph=True)

# Run the compiled function to trigger tracing
result = compiled(x, config)

# Basic assertion to verify execution
assert result is not None