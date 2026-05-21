import torch
import torch._dynamo

def test_compile_fill_diagonal_with_item():
    """
    Test case for Issue 163420.
    Verifies that torch.compile handles fill_diagonal_ correctly when the fill value
    is derived from a 0-d tensor's .item() method.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA is not available.")
        return

    # Configuration required to trigger the specific path in the bug report
    torch._dynamo.config.capture_scalar_outputs = True

    def foo(arg0, arg1):
        t0 = arg0
        t1 = arg1
        t2 = t0.clone()
        # The problematic operation: filling diagonal with a scalar extracted from a tensor
        t2.fill_diagonal_(t1.item())
        return t2

    # Setup inputs matching the bug report
    arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
    arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)

    # 1. Run Eager Mode
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    
    grad_eager_0 = arg0.grad.clone()
    grad_eager_1 = arg1.grad.clone()

    # Reset gradients for compiled run
    arg0.grad.zero_()
    arg1.grad.zero_()

    # 2. Run Compiled Mode (Inductor)
    # Using fullgraph=True and dynamic=True as per the bug report
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    
    try:
        out_compiled = compiled_foo(arg0, arg1)
        out_compiled.sum().backward()
        
        grad_compiled_0 = arg0.grad.clone()
        grad_compiled_1 = arg1.grad.clone()

        # 3. Verify Results
        assert torch.allclose(out_eager, out_compiled), "Forward pass outputs differ"
        assert torch.allclose(grad_eager_0, grad_compiled_0), "Gradients for arg0 differ"
        assert torch.allclose(grad_eager_1, grad_compiled_1), "Gradients for arg1 differ"
        
        print("Test Passed: Eager and Compiled results match.")
        
    except Exception as e:
        print(f"Test Failed with error: {e}")
        raise

if __name__ == "__main__":
    test_compile_fill_diagonal_with_item()