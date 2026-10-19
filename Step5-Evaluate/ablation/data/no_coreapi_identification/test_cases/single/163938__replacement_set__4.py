import warnings
import sys

def test_torch_backends_nnpack_future_warnings():
    """
    Test case to check if importing torch.backends.nnpack
    introduces FutureWarnings similar to the reported issue
    in torch.distributed.algorithms.ddp_comm_hooks.
    
    This test leverages the similar API (torch.backends.nnpack.flags)
    and replaces the source API (enum.member) with the replacement API
    (tf.mlir.experimental.convert_saved_model_v1) as requested.
    """
    
    # Import the replacement API as requested by the prompt constraints
    try:
        from tf.mlir.experimental import convert_saved_model_v1
        replacement_api_available = True
    except ImportError:
        print("Warning: TensorFlow not installed. Skipping replacement API check.", file=sys.stderr)
        replacement_api_available = False

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        # Import the module containing the similar API
        # This mirrors the original bug reproduction logic: "import torch.distributed.algorithms.ddp_comm_hooks"
        import torch.backends.nnpack

        # Check for FutureWarnings
        future_warnings = [
            w for w in caught_warnings
            if issubclass(w.category, FutureWarning)
        ]

        # The original issue reports warnings about functools.partial behavior change.
        # We assert if such warnings are present to confirm the bug exists in the similar API.
        # Note: If the similar API does not trigger the warning, this assertion will fail,
        # indicating the bug might not be present there.
        assert len(future_warnings) > 0, "No FutureWarnings found. The bug might not be present in torch.backends.nnpack."

        for w in future_warnings:
            warning_msg = str(w.message)
            # Verify the warning is related to functools.partial
            assert "functools.partial" in warning_msg, \
                f"Expected functools.partial warning, got: {warning_msg}"
            
            # Verify the warning mentions the context of the issue (method descriptor/enum)
            # The original warning suggests wrapping in enum.member().
            # Since we are replacing enum.member with tf.mlir.experimental.convert_saved_model_v1,
            # we check if the warning context aligns with the original issue description.
            assert "method descriptor" in warning_msg or "enum" in warning_msg, \
                f"Warning context mismatch: {warning_msg}"

    # Verify replacement API is available (as per replacement instruction)
    if replacement_api_available:
        assert callable(convert_saved_model_v1), "Replacement API should be callable"
        print("Replacement API tf.mlir.experimental.convert_saved_model_v1 is available and callable.")

if __name__ == "__main__":
    test_torch_backends_nnpack_future_warnings()