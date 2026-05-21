import torch
import sys

def test_checkpoint_cuda_graph_compatibility():
    """
    Test case adapted from Issue 162504.
    This test verifies that torch.utils.checkpoint.checkpoint works correctly
    within a CUDA graph capture context.
    
    The structure is inspired by the similar API pattern (tf.debugging.enable_traceback_filtering),
    specifically the precondition check before executing the core logic.
    """
    # Precondition check: Ensure CUDA is available, similar to the Python version check in the TF API.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA Graphs are not available. "
            "This test requires a CUDA-enabled device."
        )

    torch.cuda.manual_seed(42)

    def fn(x):
        return x * torch.sigmoid(torch.randn(1, device="cuda"))

    # Initialize device state
    fn(torch.ones(1, device="cuda"))

    # 1. Eager Execution
    torch.cuda.manual_seed(42)
    eager_in = torch.ones(1, device="cuda", requires_grad=True)
    eager_out = torch.utils.checkpoint.checkpoint(
        fn, eager_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

    # 2. CUDA Graph Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_in = torch.ones(1, device="cuda", requires_grad=True)
        graph_out = torch.utils.checkpoint.checkpoint(
            fn, graph_in,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

    # 3. Replay and Verification
    torch.cuda.manual_seed(42)
    g.replay()
    
    assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), \
        "Mismatch in gradient outputs between eager and CUDA graph execution"
    
    print("Test passed: Checkpoint works correctly under CUDA graph capture.")

if __name__ == "__main__":
    test_checkpoint_cuda_graph_compatibility()