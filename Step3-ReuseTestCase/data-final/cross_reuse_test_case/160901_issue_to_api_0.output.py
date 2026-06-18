import torch
import unittest

# Original API Under Test: torch.autograd.Function
# Similar API Pattern: tf.test.Benchmark (builder_fn, use_xla_jit -> use_dynamo)

class MyFn(torch.autograd.Function):
    """
    Custom autograd function similar to the one in the bug report.
    The bug involves requires_grad propagation not working correctly
    when this function is traced by torch.compile (Dynamo).
    """
    @staticmethod
    def forward(ctx, x):
        # Operation from the bug report
        return torch.matmul(x, (torch.ones_like(x) * 10).t())
    
    @staticmethod
    def backward(ctx, grad_out):
        # Simplified backward pass for testing propagation
        return grad_out

def run_autograd_test(use_dynamo, builder_fn, device="cpu"):
    """
    Helper function inspired by tf.test.Benchmark pattern.
    It takes a builder function and a flag to enable/disable compilation.
    """
    # Setup input matching the bug report dimensions
    x = torch.randn(2093, 256, requires_grad=True, device=device)
    
    # Apply compilation if requested (analogous to use_xla_jit)
    if use_dynamo:
        fn = torch.compile(builder_fn)
    else:
        fn = builder_fn
        
    # Run forward pass
    out = fn(x)
    
    # Verify requires_grad propagation
    # The bug report indicates this might fail silently in Dynamo
    assert out.requires_grad, (
        f"Output requires_grad is False with use_dynamo={use_dynamo}. "
        "This indicates the bug described in Issue 160901."
    )
        
    # Run backward pass to ensure the graph is connected
    out.sum().backward()
    
    assert x.grad is not None, (
        f"Input gradient is None with use_dynamo={use_dynamo}. "
        "Gradient flow is broken."
    )

class TestAutogradFunctionDynamo(unittest.TestCase):
    def test_requires_grad_propagation(self):
        """
        Test that requires_grad is correctly propagated through a custom
        autograd.Function when using torch.compile.
        """
        # The builder function representing the graph logic
        def my_graph(x):
            return MyFn.apply(x)
        
        # Test without compilation (Eager mode) - Baseline
        run_autograd_test(use_dynamo=False, builder_fn=my_graph)
        
        # Test with compilation (Dynamo mode) - Reproduces the bug
        # If the bug is present, this assertion will fail
        run_autograd_test(use_dynamo=True, builder_fn=my_graph)

if __name__ == "__main__":
    unittest.main()