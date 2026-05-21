import torch

def test_checkpoint_in_cuda_graph():
    """
    Test case to verify that torch.utils.checkpoint.checkpoint works correctly
    when captured in a CUDA graph, following the pattern of setting state,
    executing, and resetting state similar to the usage of 
    tf.distribute.experimental_set_strategy.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Set seed for reproducibility (analogous to setting strategy)
    torch.cuda.manual_seed(42)

    def fn(x):
        # A function involving RNG to test state preservation
        return x * torch.sigmoid(torch.randn(1, device="cuda"))

    # Initialize device state
    fn(torch.ones(1, device="cuda"))

    # --- Eager Execution ---
    # Set state
    torch.cuda.manual_seed(42)
    eager_in = torch.ones(1, device="cuda", requires_grad=True)
    eager_out = torch.utils.checkpoint.checkpoint(
        fn, eager_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

    # --- CUDA Graph Capture ---
    g = torch.cuda.CUDAGraph()
    graph_in = torch.ones(1, device="cuda", requires_grad=True)
    
    # Capture the forward and backward pass within the graph scope
    with torch.cuda.graph(g):
        graph_out = torch.utils.checkpoint.checkpoint(
            fn, graph_in,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

    # --- Replay ---
    # Reset state before replay to ensure consistency
    torch.cuda.manual_seed(42)
    g.replay()

    # --- Verification ---
    # Assert that the graph replay matches the eager execution
    assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), \
        f"Mismatch in gradient outputs: eager={eager_in_grad}, graph={graph_in_grad}"

    print("Test passed: Checkpoint behavior is consistent under CUDA graph capture.")

if __name__ == "__main__":
    test_checkpoint_in_cuda_graph()