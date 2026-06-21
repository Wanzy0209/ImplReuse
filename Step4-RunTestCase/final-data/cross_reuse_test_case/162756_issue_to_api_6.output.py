import torch
import torch.distributed as dist

def test_compile_combo_kernels_with_cumsum():
    """
    Test case for Issue #162756.
    Verifies that torch.compile works with combo_kernels enabled
    when operations requiring helper functions (like cumsum) are used.
    
    This test leverages torch.distributed.is_available (the similar API)
    to ensure the test runs in a compatible environment, and reuses the
    hasattr pattern from that API's implementation to safely check
    for the combo_kernels configuration.
    """
    
    # Leverage the similar API to check environment compatibility
    if not torch.distributed.is_available():
        print("Skipping test: torch.distributed is not available.")
        return

    # Reuse the hasattr pattern from the similar API implementation
    # (return hasattr(torch._C, "_c10d_init")) to safely access the inductor config.
    # First, check if torch._inductor exists
    if not hasattr(torch, "_inductor"):
        print("Skipping test: torch._inductor is not available.")
        return

    # Check if config exists
    if not hasattr(torch._inductor, "config"):
        print("Skipping test: torch._inductor.config is not available.")
        return

    # Check if combo_kernels exists
    if hasattr(torch._inductor.config, "combo_kernels"):
        torch._inductor.config.combo_kernels = True
    else:
        print("Skipping test: combo_kernels config not found.")
        return

    # The original bug requires CUDA
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available.")
        return

    @torch.compile
    def fn(x, y, z):
        return x.sum(1), y.mean(1), z.cumsum(1)

    inps = (
        torch.rand(16, 128, device="cuda"),
        torch.rand(32, 128, device="cuda"),
        torch.rand(32, 256, device="cuda"),
    )

    # Run the compiled function.
    # Bug reproduction: This previously raised NameError: _triton_helper_fn_add0 is not defined
    try:
        out = fn(*inps)
    except NameError as e:
        print(f"Bug reproduced: {e}")
        raise

    # Verify correctness against eager execution
    expected = (
        inps[0].sum(1),
        inps[1].mean(1),
        inps[2].cumsum(1)
    )

    for o, e in zip(out, expected):
        assert torch.allclose(o, e), "Output mismatch between compiled and eager execution"

if __name__ == "__main__":
    test_compile_combo_kernels_with_cumsum()