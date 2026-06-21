import torch
import pytest

def test_compile_cumsum_combo_kernels_with_sdp_config():
    """
    Test that torch.compile works with combo_kernels enabled and cumsum,
    verifying behavior across different torch.backends.cuda.math_sdp_enabled states.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Leverage the similar API: torch.backends.cuda.math_sdp_enabled
    # We check the current state and toggle it to ensure the bug fix is robust
    # against different backend math configurations.
    original_sdp_state = torch.backends.cuda.math_sdp_enabled()

    try:
        for sdp_enabled in [True, False]:
            # Set the SDP math state
            torch.backends.cuda.enable_math_sdp(sdp_enabled)
            
            # Verify the state using the similar API
            assert torch.backends.cuda.math_sdp_enabled() == sdp_enabled

            # Original Bug Reproduction Logic
            torch._inductor.config.combo_kernels = True

            @torch.compile
            def fn(x, y, z):
                return x.sum(1), y.mean(1), z.cumsum(1)

            inps = (
                torch.rand(16, 128, device="cuda"),
                torch.rand(32, 128, device="cuda"),
                torch.rand(32, 256, device="cuda"),
            )

            # The bug was a NameError for '_triton_helper_fn_add0'.
            # We assert that the function executes successfully without raising NameError.
            try:
                result = fn(*inps)
                assert result is not None
            except NameError as e:
                pytest.fail(f"NameError raised with combo_kernels=True and math_sdp_enabled={sdp_enabled}: {e}")

    finally:
        # Restore original state
        torch.backends.cuda.enable_math_sdp(original_sdp_state)