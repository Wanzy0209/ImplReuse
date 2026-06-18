import torch

# Reuse pattern: Check pre-conditions (similar to sys.version_info check in tf.keras.config.enable_traceback_filtering)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This test requires a GPU.")

torch.cuda.manual_seed(42)

def fn(x):
    return x * torch.sigmoid(torch.randn(1, device="cuda"))

# Initialize device state
fn(torch.ones(1, device="cuda"))

# Reuse pattern: Iterate over configuration states (similar to toggling global flags in the similar API)
# to ensure the fix works across different checkpoint configurations.
for preserve_rng_state in [True, False]:
    print(f"Testing with preserve_rng_state={preserve_rng_state}")

    torch.cuda.manual_seed(42)
    eager_in = torch.ones(1, device="cuda", requires_grad=True)
    eager_out = torch.utils.checkpoint.checkpoint(
        fn, eager_in,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
    )
    eager_in_grad, = torch.autograd.grad(eager_out, eager_in)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_in = torch.ones(1, device="cuda", requires_grad=True)
        graph_out = torch.utils.checkpoint.checkpoint(
            fn, graph_in,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )
        graph_in_grad, = torch.autograd.grad(graph_out, graph_in)

    torch.cuda.manual_seed(42)
    g.replay()
    
    # Assert correctness
    assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0), \
        f"Mismatch in gradient outputs with preserve_rng_state={preserve_rng_state}"
    
    print(f"Passed for preserve_rng_state={preserve_rng_state}")

print("All tests passed.")