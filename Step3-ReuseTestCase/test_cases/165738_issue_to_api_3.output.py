import torch
import torch.nn.functional as F
import time

def test_sigmoid_xpu_performance():
    """
    Test case to verify performance of torch.nn.functional.sigmoid on XPU.
    This test is derived from Issue #165738 where tl.sqrt_rn caused a 500x 
    slowdown on Intel XPU. We check if the similar API (sigmoid) suffers 
    from similar performance regressions when run via Inductor/Triton on XPU.
    """
    # Check for XPU availability
    if not torch.xpu.is_available():
        print("XPU not available. Skipping test.")
        return

    device = torch.device("xpu")
    
    # Define the function using the similar API: torch.nn.functional.sigmoid
    def sigmoid_op(x):
        return F.sigmoid(x)

    # Compile the function to trigger Inductor/Triton generation, 
    # mirroring the context of the original bug where Inductor generated kernels.
    compiled_sigmoid = torch.compile(sigmoid_op, backend="inductor")

    # Setup input data similar to the bug report (fp16, large size)
    # The bug report hints at size ~1GB. We use a smaller size for testability 
    # but large enough to measure throughput (64M elements).
    size = 1024 * 1024 * 64 
    x = torch.randn(size, dtype=torch.float16, device=device)

    # Warmup run to ensure compilation is done and caches are warm
    _ = compiled_sigmoid(x)
    torch.xpu.synchronize()

    # Performance measurement
    start = time.time()
    iterations = 10
    for _ in range(iterations):
        _ = compiled_sigmoid(x)
    torch.xpu.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations
    print(f"Average execution time for sigmoid on XPU: {avg_time:.6f}s")

    # The original bug reported a 500x slowdown. 
    # We assert that the operation completes within a reasonable threshold 
    # to catch similar performance regressions.
    # Threshold: 1.0 second is a generous upper bound for this operation size on a modern GPU/XPU.
    assert avg_time < 1.0, f"Performance regression detected: sigmoid took {avg_time:.6f}s, expected < 1.0s"

if __name__ == "__main__":
    test_sigmoid_xpu_performance()