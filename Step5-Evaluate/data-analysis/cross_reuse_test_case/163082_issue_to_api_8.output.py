import torch
import torch.nn.functional as F

def test_local_response_norm_compile_cuda():
    """
    Test case adapted from Issue 163082.
    Original Issue: torch.nn.functional.normalize outputs vectors with norm>1 
    with torch.compile + cuda.
    
    This test applies the same reproduction logic to the similar API 
    torch.nn.functional.local_response_norm to check for numerical 
    discrepancies between compiled and eager execution on CUDA.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch 2.0+), skipping test.")
        return

    torch.manual_seed(1337)

    # Define the compiled function for the similar API
    @torch.compile()
    def lrn_compiled(input_tensor):
        return F.local_response_norm(input_tensor, size=5)

    # Define the eager function for the similar API
    def lrn_eager(input_tensor):
        return F.local_response_norm(input_tensor, size=5)

    device = 'cuda'
    
    # Create a 4D tensor (Batch, Channel, Height, Width) as required by local_response_norm.
    # We use the specific float value from the original issue (3.799999) to attempt to 
    # trigger similar floating-point precision edge cases.
    # Shape: (1, 1, 1, 3)
    c = torch.tensor([[[[3.799999, 0.0, 0.0]]]], device=device, dtype=torch.float32)

    print("Input tensor:", c)

    # Run compiled version
    res_compiled = lrn_compiled(c)
    print("Output (compiled):", res_compiled)

    # Run eager version
    res_eager = lrn_eager(c)
    print("Output (eager):", res_eager)

    # Check for numerical consistency.
    # The original bug manifested as a norm > 1.0 (approx 1.0000001192092896).
    # We use a strict tolerance to detect similar precision drifts.
    # atol=1e-7 is chosen because the original error was on the order of 1.19e-7.
    assert torch.allclose(res_compiled, res_eager, rtol=1e-5, atol=1e-7), \
        f"Mismatch between compiled and eager execution. Max diff: {torch.max(torch.abs(res_compiled - res_eager))}"

if __name__ == "__main__":
    test_local_response_norm_compile_cuda()