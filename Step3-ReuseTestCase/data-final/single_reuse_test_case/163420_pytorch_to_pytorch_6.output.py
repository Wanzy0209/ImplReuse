import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def foo(A):
    # Using torch.lobpcg as the similar API
    # lobpcg computes eigenvalues and eigenvectors of a symmetric positive definite matrix
    eigenvalues, eigenvectors = torch.lobpcg(A)
    return eigenvalues, eigenvectors

if __name__ == '__main__':
    # Setup inputs
    # Create a symmetric positive definite matrix required by lobpcg
    # Using CUDA to match the original bug's device context
    torch.manual_seed(42)
    A = torch.randn(5, 5, dtype=torch.float32, device='cuda')
    A = A @ A.T + 1e-3 * torch.eye(5, device='cuda')
    A.requires_grad = True

    print("Running Eager Mode...")
    try:
        out_eager = foo(A)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')
        out_eager = None

    print("\nRunning Compiled Mode (fullgraph=True, dynamic=True)...")
    try:
        # Using fullgraph and dynamic flags as in the original bug report
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(A)
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
        out_compiled = None

    # Verify results if both succeeded
    if out_eager is not None and out_compiled is not None:
        # lobpcg returns a tuple of (eigenvalues, eigenvectors)
        e_eager, v_eager = out_eager
        e_comp, v_comp = out_compiled
        
        # Check closeness of eigenvalues
        assert torch.allclose(e_eager, e_comp, atol=1e-4), "Eigenvalues mismatch"
        # Check closeness of eigenvectors (accounting for sign ambiguity)
        assert torch.allclose(torch.abs(v_eager), torch.abs(v_comp), atol=1e-4), "Eigenvectors mismatch"
        print("\nVerification Passed: Eager and Compiled outputs match.")