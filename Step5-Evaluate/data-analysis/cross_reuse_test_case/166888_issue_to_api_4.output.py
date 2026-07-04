import torch
import pytest

# The bug is specific to the CUDA/Inductor backend
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

def test_inductor_item_on_float_tensor_arg():
    """
    Regression test for Issue 166888.
    
    Verifies that calling .item() on a float tensor argument inside a function
    compiled with torch.compile (backend='inductor', fullgraph=True) works correctly
    and does not raise a NameError in the generated Triton kernel.
    """
    def f(x, max_val):
        # The bug occurs when .item() is called on a tensor argument (max_val)
        # and the result is used in an operation like torch.clamp.
        y = torch.clamp(x, 0, max_val.item())
        return y

    # Compile with the specific settings that triggered the bug
    compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

    # Setup inputs
    x = torch.randn(10, 20, 30, device='cuda')
    max_val = torch.tensor(5.0, device='cuda')

    # Run the compiled function
    # If the bug is present, this will raise:
    # torch._inductor.exc.InductorError: NameError: 'zuf0' is not defined
    result = compiled_func(x, max_val)

    # Verify the output is numerically correct
    expected = torch.clamp(x, 0, 5.0)
    assert torch.allclose(result, expected)

if __name__ == "__main__":
    test_inductor_item_on_float_tensor_arg()
    print("Test passed.")