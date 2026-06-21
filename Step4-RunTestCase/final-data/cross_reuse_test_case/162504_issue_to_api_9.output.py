import unittest
import torch
import torch.utils.checkpoint  # Explicitly import to ensure the submodule is available

class TestCheckpointCUDAGraph(unittest.TestCase):
    """
    Test case to verify torch.utils.checkpoint.checkpoint works under CUDA graph capture.
    Based on Issue ID: 162504.
    """

    def _is_cuda_supported(self):
        """
        Helper to check for CUDA availability, mimicking the pattern of 
        tf.test.is_built_with_xla for feature detection.
        """
        return torch.cuda.is_available()

    def test_checkpoint_in_cuda_graph(self):
        # Leverage the similar API pattern: Check for feature support before running
        if not self._is_cuda_supported():
            self.skipTest("CUDA is not available, skipping CUDA graph test.")

        # Check if checkpoint is available (handles cases where the submodule might not be loaded)
        if not hasattr(torch.utils, 'checkpoint'):
            self.skipTest("torch.utils.checkpoint is not available, skipping test.")

        torch.cuda.manual_seed(42)

        def fn(x):
            return x * torch.sigmoid(torch.randn(1, device="cuda"))

        # initialize device state
        fn(torch.ones(1, device="cuda"))

        torch.cuda.manual_seed(42)
        eager_in = torch.ones(1, device="cuda", requires_grad=True)
        eager_out = torch.utils.checkpoint.checkpoint(
            fn, eager_in,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_in = torch.ones(1, device="cuda", requires_grad=True)
            graph_out = torch.utils.checkpoint.checkpoint(
                fn, graph_in,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

        torch.cuda.manual_seed(42)
        g.replay()
        
        # Assert that the graph replay matches the eager execution
        self.assertTrue(torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), 
                        "Mismatch in gradient outputs between eager and CUDA graph execution")

if __name__ == '__main__':
    unittest.main()