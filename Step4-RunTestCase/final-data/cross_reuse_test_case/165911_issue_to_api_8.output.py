import torch
import unittest

# Check for the availability of the internal torch._dynamo API
# This prevents AttributeError in environments where torch._dynamo is not exposed
HAS_DYNAMO = hasattr(torch, '_dynamo')

class TestDynamoGraphCaptureWithLog1p(unittest.TestCase):
    """
    Test case for Issue 165911: Incorrect user code stack when using 
    torch._dynamo.functional_export._dynamo_graph_capture_for_export.
    
    This test adapts the original reproduction logic by incorporating 
    the semantic equivalent of the similar API (tf.compat.v1.math.log1p),
    which translates to torch.log1p in PyTorch.
    """

    def compute(self, x, w):
        """
        Modified compute function to leverage the similar API pattern (log1p).
        Original API: tf.compat.v1.math.log1p
        PyTorch Equivalent: torch.log1p
        """
        linear_out = torch.nn.functional.linear(x, w)
        # Applying log1p as seen in the similar API usage pattern
        return torch.log1p(linear_out)

    def nop(self, x, w):
        torch._check(x.shape[0] == 0)
        return torch.empty_like(x)

    def chunked_compute(self, x, w):
        sz = x.shape[0]
        torch._check(sz <= 8)
        # Using torch.cond which is central to the reported bug
        out0 = torch.cond(sz > 0, self.compute, self.nop, (x[0:2], w))
        out1 = torch.cond(sz > 2, self.compute, self.nop, (x[2:4], w))
        out2 = torch.cond(sz > 4, self.compute, self.nop, (x[4:6], w))
        out3 = torch.cond(sz > 6, self.compute, self.nop, (x[6:8], w))
        return torch.cat([out0, out1, out2, out3])

    @unittest.skipIf(not HAS_DYNAMO, "torch._dynamo is not available in this environment")
    def test_dynamo_capture_stack_trace(self):
        class Model(torch.nn.Module):
            def __init__(self, compute_func, chunked_compute_func):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.compute = compute_func
                self.chunked_compute = chunked_compute_func

            def forward(self, x):
                # Note: Using .weight instead of .w to ensure valid PyTorch API usage
                return self.chunked_compute(x, self.linear.weight)

        # Setup inputs
        x = torch.randn(4, 16, requires_grad=True)
        
        # Instantiate model with bound methods
        model = Model(self.compute, self.chunked_compute)

        # The API under test
        # This is expected to trigger the "Incorrect user code stack" bug
        try:
            mod = torch._dynamo.functional_export._dynamo_graph_capture_for_export(model)(x)
            # If the bug is fixed, this should complete without error or stack trace issues.
            self.assertIsNotNone(mod)
        except Exception as e:
            # Catching the error to report it in the test output
            self.fail(f"torch._dynamo.functional_export._dynamo_graph_capture_for_export failed with: {e}")

if __name__ == "__main__":
    unittest.main()