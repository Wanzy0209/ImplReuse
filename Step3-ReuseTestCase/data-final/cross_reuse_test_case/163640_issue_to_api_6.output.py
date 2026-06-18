import torch

def test_nested_tensor_from_mask():
    """
    Test case for torch._nested_tensor_from_mask_left_aligned.
    
    This test is derived from Issue 163640, where compiling a TransformerEncoder
    with a boolean src_key_padding_mask fails under Dynamo fullgraph mode.
    The root cause is identified as torch._nested_tensor_from_mask_left_aligned
    returning a bool instead of a Tensor.
    
    The test structure mimics the direct wrapper pattern seen in the similar API
    (tf.compat.v1.user_ops.my_fact) to isolate the internal function call.
    """
    # Setup inputs matching the bug report
    B, T = 1, 41
    mask = (torch.rand(B, T) > 0.5)
    mask[..., 0] = True

    # Define a wrapper function that calls the specific internal API.
    # This mimics the direct call pattern of the similar API (my_fact).
    def call_internal_api(mask):
        # We pass dummy tensors for the other required arguments (tensor, offset)
        # to focus the test on the mask handling behavior.
        dummy_tensor = torch.empty(0)
        dummy_offset = torch.empty(0, dtype=torch.int64)
        return torch._nested_tensor_from_mask_left_aligned(
            mask, dummy_tensor, dummy_offset
        )

    # 1. Verify Eager Mode behavior
    # The bug report states eager mode works correctly.
    try:
        res_eager = call_internal_api(mask)
        assert isinstance(res_eager, torch.Tensor), \
            f"Eager mode failed: expected Tensor, got {type(res_eager)}"
    except Exception as e:
        print(f"Eager mode failed unexpectedly: {e}")
        raise

    # 2. Verify Compiled Mode (fullgraph=True)
    # The bug report states this mode fails because the internal call returns a bool.
    try:
        compiled_fn = torch.compile(call_internal_api, fullgraph=True)
        res_compiled = compiled_fn(mask)
        
        # The core assertion: the function must return a Tensor, not a bool.
        assert isinstance(res_compiled, torch.Tensor), \
            f"Compiled mode failed: expected Tensor, got {type(res_compiled)}. " \
            "This indicates the bug where a bool is returned instead of a Tensor."
            
    except torch._dynamo.exc.Unsupported as e:
        # This exception is expected if the bug is present (returning bool causes graph break)
        raise AssertionError(
            "Bug reproduced: torch._dynamo.exc.Unsupported raised. "
            "This likely happens because torch._nested_tensor_from_mask_left_aligned "
            "returned a bool instead of a Tensor."
        ) from e

if __name__ == "__main__":
    test_nested_tensor_from_mask()
    print("Test passed successfully.")