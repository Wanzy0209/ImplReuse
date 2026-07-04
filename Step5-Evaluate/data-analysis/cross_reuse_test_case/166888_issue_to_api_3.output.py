import torch
import pytest

# Test case for Issue 166888: NameError when using .item() on float tensor arg in torch.compile
# The bug occurs when a scalar value extracted via .item() from a tensor argument
# is used inside a compiled function (e.g., as a bound for clamp).
# This relates to the pattern of handling scalar values (similar to aggregation logic in other frameworks).

def test_compile_item_float_tensor_arg():
    """
    Tests that torch.compile handles .item() on float tensor arguments correctly
    without raising NameError in the generated Triton kernel.
    """
    # Check if CUDA is available as the bug is specific to CUDA/Inductor
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Inductor test")

    def f(x, max_val):
        # The bug is triggered here: using .item() on a tensor argument
        # inside a function compiled with fullgraph=True
        y = torch.clamp(x, 0, max_val.item())
        return y

    # Compile with inductor backend and fullgraph=True (as per the bug report)
    compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

    # Setup inputs
    x = torch.randn(10, 20, 30, device='cuda')
    max_val = torch.tensor(5.0, device='cuda')

    # Run the compiled function
    # This should not raise torch._inductor.exc.InductorError: NameError: 'zuf0' is not defined
    try:
        result = compiled_func(x, max_val)
    except Exception as e:
        pytest.fail(f"Inductor compilation failed with error: {e}")

    # Verify correctness against eager execution
    expected = f(x, max_val)
    assert torch.allclose(result, expected), "Compiled output does not match eager output"

def test_compile_item_int_tensor_arg():
    """
    Additional test to ensure robustness of scalar handling with integer tensors.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Inductor test")

    def g(x, multiplier):
        return x * multiplier.item()

    compiled_g = torch.compile(g, backend='inductor', fullgraph=True)
    x = torch.randn(5, 5, device='cuda')
    multiplier = torch.tensor(2, device='cuda')

    try:
        result = compiled_g(x, multiplier)
    except Exception as e:
        pytest.fail(f"Inductor compilation failed with error: {e}")

    expected = g(x, multiplier)
    assert torch.allclose(result, expected)