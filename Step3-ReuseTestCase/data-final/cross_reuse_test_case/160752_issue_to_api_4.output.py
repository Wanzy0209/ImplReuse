import torch
import torch.nn.functional as F
from torch import func as torch_func

def test_jacfwd_one_hot_dynamic_compile():
    """
    Test case for Issue 160752.
    Verifies that torch.compile with dynamic=True works correctly when
    the function involves torch.func.jacfwd and torch.nn.functional.one_hot.
    
    The bug report indicates a failure when dynamic=True is enabled.
    This test ensures the compiled function produces the same output
    as the eager execution.
    """
    MAX = 3
    BATCH = 37

    def func(x, idxs):
        return x.square() * F.one_hot(idxs, MAX)

    def jacfunc(x, idxs):
        return torch_func.jacfwd(func, argnums=(0,))(x, idxs)

    # Setup inputs
    idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
    x = torch.rand((BATCH, MAX), dtype=torch.float64)

    # 1. Baseline: Eager execution
    expected_out = jacfunc(x, idxs)

    # 2. Target: Compiled execution with dynamic=True
    # This configuration was reported to fail.
    compiled_jacfunc = torch.compile(jacfunc, dynamic=True)
    
    # Execute the compiled function
    actual_out = compiled_jacfunc(x, idxs)

    # 3. Validation: Check if outputs match
    assert torch.allclose(expected_out, actual_out, atol=1e-5), \
        "Output mismatch between eager and compiled execution"

if __name__ == "__main__":
    test_jacfwd_one_hot_dynamic_compile()
    print("Test passed successfully.")