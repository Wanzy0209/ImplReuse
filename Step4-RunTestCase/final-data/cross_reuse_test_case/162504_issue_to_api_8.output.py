import torch
import torch.utils.checkpoint  # Explicitly import to fix AttributeError

def test_checkpoint_cuda_graph_state_preservation():
    """
    Test case to verify torch.utils.checkpoint.checkpoint works correctly
    within a CUDA graph capture, specifically regarding RNG state preservation.
    
    This test mirrors the pattern of tf.keras.backend.get_uid, which manages
    state (UIDs) scoped to the current graph context. Here, we verify that
    PyTorch's checkpoint mechanism correctly handles RNG state within the
    torch.cuda.graph context, ensuring consistency with eager execution.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    torch.cuda.manual_seed(42)

    # A function that uses RNG, making it state-dependent
    def fn(x):
        return x * torch.sigmoid(torch.randn(1, device="cuda"))

    # Initialize device state (warmup)
    fn(torch.ones(1, device="cuda"))

    # --- Eager Execution (Baseline) ---
    torch.cuda.manual_seed(42)
    eager_in = torch.ones(1, device="cuda", requires_grad=True)
    eager_out = torch.utils.checkpoint.checkpoint(
        fn, eager_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

    # --- CUDA Graph Execution ---
    # Capture the graph. Similar to tf.keras.backend.get_uid identifying the graph,
    # we enter a specific graph context here.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_in = torch.ones(1, device="cuda", requires_grad=True)
        graph_out = torch.utils.checkpoint.checkpoint(
            fn, graph_in,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

    # Replay the graph
    torch.cuda.manual_seed(42)
    g.replay()

    # Verify that the state (RNG) was handled correctly in the graph context.
    # If checkpoint fails to handle state correctly under graph capture,
    # the gradients will mismatch.
    assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), \
        f"Mismatch in gradient outputs:\nEager: {eager_in_grad}\nGraph: {graph_in_grad}"
    
    print("Test passed: Gradients match between eager and CUDA graph execution.")

if __name__ == "__main__":
    test_checkpoint_cuda_graph_state_preservation()