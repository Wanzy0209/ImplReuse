import torch

# This test case is derived from the bug report (Issue 167369) where torch.compile
# fails to trace the builtin `repr` operator on user-defined objects.
# It leverages the pattern from the similar API (tf.keras.constraints.serialize)
# by defining a 'Constraint' class and attempting to serialize it (via repr)
# within the compiled graph.

class Constraint:
    """
    Mimics a Keras constraint object.
    The similar API (tf.keras.constraints.serialize) deals with serializing
    such objects. Here we use the builtin repr() to trigger the serialization
    logic within the Dynamo tracer.
    """
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return f"Constraint(min_val={self.min_val}, max_val={self.max_val})"


def forward(x, constraint):
    """
    Forward pass that attempts to serialize the constraint object using repr().
    This mirrors the behavior of tf.keras.constraints.serialize which converts
    an object to a string representation, but triggers the Dynamo bug in PyTorch.
    """
    # Calling repr() on non-constant user object triggers the bug
    serialized_str = repr(constraint)
    return x * len(serialized_str)


def test_compile_trace_repr():
    """
    Test that torch.compile can handle repr() calls on user-defined objects.
    """
    constraint = Constraint(min_val=-1.0, max_val=1.0)
    x = torch.randn(2, 2)

    # Compile with fullgraph=True to ensure strict tracing
    compiled = torch.compile(forward, fullgraph=True)
    
    # Execute the compiled function
    result = compiled(x, constraint)
    
    # Assertions to verify correctness
    expected_len = len(repr(constraint))
    expected_result = x * expected_len
    
    assert result.shape == (2, 2), "Output shape mismatch"
    assert torch.allclose(result, expected_result), "Output value mismatch"


if __name__ == "__main__":
    test_compile_trace_repr()