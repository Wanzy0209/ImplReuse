import torch

# Reusing the pattern from torch.utils.cpp_extension.verify_ninja_availability
# to ensure the environment supports the feature being tested.
def is_dynamic_compile_available():
    """Return True if torch.compile with dynamic shapes is available."""
    try:
        @torch.compile(dynamic=True)
        def dummy(y):
            return y + 1
        dummy(torch.zeros(1))
        return True
    except Exception:
        return False

def verify_dynamic_compile_availability():
    """Raise RuntimeError if torch.compile with dynamic shapes is not available."""
    if not is_dynamic_compile_available():
        raise RuntimeError("torch.compile with dynamic=True is required for this test")

def test_fill_diagonal_dynamic_shapes():
    # Verify environment first (pattern reuse from similar API)
    verify_dynamic_compile_availability()

    # Original bug reproduction logic
    @torch.compile(dynamic=True)
    def f(x):
        x.fill_diagonal_(True)

    x = torch.zeros(4, 4)
    
    # This should not raise torch._dynamo.exc.TorchRuntimeError
    f(x)

    # Assertions to verify the operation worked correctly
    assert x[0, 0] == 1, "Diagonal element (0,0) should be True (1)"
    assert x[1, 1] == 1, "Diagonal element (1,1) should be True (1)"
    assert x[2, 2] == 1, "Diagonal element (2,2) should be True (1)"
    assert x[3, 3] == 1, "Diagonal element (3,3) should be True (1)"
    assert x[0, 1] == 0, "Non-diagonal element (0,1) should remain False (0)"

if __name__ == "__main__":
    test_fill_diagonal_dynamic_shapes()
    print("Test passed.")