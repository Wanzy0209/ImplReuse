import torch
import sys

# Replicate the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(A, k):
    # Using torch.lobpcg as the target API
    # lobpcg finds the k largest eigenvalues and corresponding eigenvectors of A
    e, v = torch.lobpcg(A, k=k)
    return e, v

# Setup inputs
# Note: lobpcg is computationally intensive (O(N^3)), so we use a smaller size 
# than the original bug report to ensure the test runs in reasonable time,
# but large enough to potentially trigger memory issues similar to the bug.
N = 512
k = 5

# Create a symmetric positive definite matrix A
A = torch.randn(N, N, dtype=torch.float32, device='cuda')
A = A @ A.T + 1.0 * torch.eye(N, device='cuda')
A.requires_grad = True

if __name__ == '__main__':
    # Eager execution
    try:
        out_eager = foo(A, k)
        out_eager[0].sum().backward()
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')
        sys.exit(1)

    # Compiled execution
    try:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(A, k)
        out_compiled[0].sum().backward()
        print('Compile Success! ')
        
        # Verify correctness
        assert torch.allclose(out_eager[0], out_compiled[0], atol=1e-3)
        print('Results match! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
        sys.exit(1)