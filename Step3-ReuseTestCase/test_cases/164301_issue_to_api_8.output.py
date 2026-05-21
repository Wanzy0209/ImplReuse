import torch
import time

def test_scaled_modified_bessel_k1_compile_regression():
    """
    Test case adapted from Issue 164301 logic.
    
    Original Issue: torch.compile regression in mxfp8 quantization (dim0).
    Adaptation: Verify that torch.compile does not introduce a severe performance 
    regression for torch.special.scaled_modified_bessel_k1 on large tensors.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    device = "cuda"
    
    # Mimic the large scale of the original bug (M=16384, K=16384)
    # to ensure the inductor codegen is sufficiently stressed.
    N = 16384 * 16384
    x = torch.randn(N, device=device, dtype=torch.float32)

    # Define the function using the similar API
    def bessel_op(x):
        return torch.special.scaled_modified_bessel_k1(x)

    # Compile the function using torch.compile (the API under test in the issue)
    compiled_bessel_op = torch.compile(bessel_op)

    # Warmup runs
    _ = bessel_op(x)
    _ = compiled_bessel_op(x)
    torch.cuda.synchronize()

    # Benchmark Eager execution
    start = time.time()
    for _ in range(10):
        res_eager = bessel_op(x)
    torch.cuda.synchronize()
    eager_time = time.time() - start

    # Benchmark Compiled execution
    start = time.time()
    for _ in range(10):
        res_compiled = compiled_bessel_op(x)
    torch.cuda.synchronize()
    compiled_time = time.time() - start

    # 1. Correctness Assertion
    # Ensure the compiled kernel produces the same results as eager
    assert torch.allclose(res_eager, res_compiled, atol=1e-5), \
        "Compiled output differs from eager output"

    # 2. Performance Regression Assertion
    # The original bug reported a ~4x performance degradation (5600gbps -> 1485gbps).
    # We assert that the compiled version is not significantly slower than eager.
    # (Ideally it should be faster, but we guard against severe regressions).
    slowdown_factor = compiled_time / eager_time
    
    print(f"Eager time: {eager_time:.4f}s")
    print(f"Compiled time: {compiled_time:.4f}s")
    print(f"Slowdown factor: {slowdown_factor:.2f}x")

    # If compiled is > 3x slower, it indicates a potential regression similar to the issue.
    assert slowdown_factor < 3.0, \
        f"Performance regression detected: compiled is {slowdown_factor:.2f}x slower than eager"

if __name__ == "__main__":
    test_scaled_modified_bessel_k1_compile_regression()