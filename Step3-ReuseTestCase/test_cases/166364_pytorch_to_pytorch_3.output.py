import torch
import torch.nn as nn
from torch.library import Library, impl, impl_abstract

# Define a custom library and operator to mimic the behavior in the bug report
lib = Library("test_flex_lib", "DEF")
lib.define("score_mod_op(Tensor score, Tensor scalar) -> Tensor")

# Use the similar API: torch.library.impl_abstract
# This defines the behavior for FakeTensors (meta kernel)
@impl_abstract("test_flex_lib::score_mod_op")
def score_mod_abstract(score, scalar):
    # The abstract implementation must return a tensor with the correct shape.
    # Since we are adding a scalar, the shape remains the same as 'score'.
    return score

# Register the concrete implementation
@impl("test_flex_lib::score_mod_op", "CPU")
def score_mod_impl(score, scalar):
    # This mimics the logic from the bug report: score = score + temp
    return score + scalar

def test_custom_op_with_scalar(use_compile=False):
    """
    Test function to verify the custom operator works with learnable scalars.
    This tests both forward and backward passes to catch compilation errors,
    mirroring the original bug report.
    """
    # Learnable scalar (from the original bug report)
    temp = nn.Parameter(torch.tensor(0.0))

    # Input tensor representing 'score'
    score = torch.randn(2, 2, requires_grad=True)

    # Function using the custom operator
    def run_op(score):
        return torch.ops.test_flex_lib.score_mod_op(score, temp)

    if use_compile:
        run_op = torch.compile(run_op)

    # Forward pass
    output = run_op(score)

    # Backward pass
    output.sum().backward()

    # Assertions to ensure execution
    assert output is not None
    assert score.grad is not None
    assert temp.grad is not None

if __name__ == "__main__":
    print("Testing without compile...")
    test_custom_op_with_scalar(use_compile=False)
    print("Testing with compile...")
    test_custom_op_with_scalar(use_compile=True)
    print("All tests passed.")