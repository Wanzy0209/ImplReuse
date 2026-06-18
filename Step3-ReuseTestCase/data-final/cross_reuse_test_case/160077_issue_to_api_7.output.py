import torch

def test_broadcast_to_compile_device_context():
    """
    Test case for torch.broadcast_to under torch.compile inside a torch.device context.
    This test is derived from Issue 160077 which reported an AttributeError for torch.split
    due to incorrect graph generation (torch._tensor.split).
    We verify that torch.broadcast_to does not suffer from the same issue.
    """
    
    # Define the function using the similar API: torch.broadcast_to
    def f(xs):
        # Original issue used: return xs.split(1, dim=0)
        # Adapted to use: torch.broadcast_to
        return torch.broadcast_to(xs, (2, 2, 2))

    # Custom backend to inspect the graph, mimicking the issue's repro
    def backend(gm, inps):
        gm.print_readable()
        return gm

    # Determine device. The original bug was specific to CUDA context.
    # We use CUDA if available to strictly reproduce the environment,
    # otherwise fallback to CPU to ensure the test is runnable on all machines.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running test on device: {device_type}")

    with torch.device(device_type):
        xs = torch.randn(2, 2, device=device_type)

        # 1. Test Eager execution (Baseline)
        eager_result = f(xs)
        print("Eager execution successful.")

        # 2. Test Compiled execution
        # The original bug triggered: `module 'torch._tensor' has no attribute 'split'`
        # We check if torch.broadcast_to triggers a similar AttributeError.
        try:
            compiled_result = torch.compile(f, backend=backend)(xs)
            print("Compiled execution successful.")
            
            # Verify results match
            assert torch.allclose(eager_result, compiled_result), "Results mismatch between eager and compiled"
            print("Test passed: Results match and no AttributeError occurred.")
            
        except AttributeError as e:
            if "torch._tensor" in str(e):
                print(f"Bug reproduced for torch.broadcast_to: {e}")
                raise
            else:
                # Re-raise if it's a different AttributeError
                raise

if __name__ == "__main__":
    test_broadcast_to_compile_device_context()