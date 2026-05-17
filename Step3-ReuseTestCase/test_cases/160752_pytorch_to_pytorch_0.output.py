import torch
import torch.nn.functional as F

def test_compile_jacfwd_one_hot_dynamic():
    """
    Test case for Issue 160752: torch.func.jacfwd fails with one_hot and torch.compile(..., dynamic=True)
    """
    MAX = 3
    BATCH = 37

    def func(x, idxs):
        return x.square() * F.one_hot(idxs, MAX)

    def jacfunc(x, idxs):
        return torch.func.jacfwd(func, argnums=(0,))(x, idxs)

    # Setup inputs
    idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
    x = torch.rand((BATCH, MAX), dtype=torch.float64)

    # 1. Run uncompiled version to get expected result
    expected_out = jacfunc(x, idxs)

    # 2. Run compiled version with dynamic=True (the failing case in the bug report)
    compiled_jacfunc = torch.compile(jacfunc, dynamic=True)
    actual_out = compiled_jacfunc(x, idxs)

    # 3. Verify that the compiled output matches the expected output
    assert expected_out.shape == actual_out.shape, \
        f"Shape mismatch: expected {expected_out.shape}, got {actual_out.shape}"
    
    assert torch.allclose(expected_out, actual_out, atol=1e-5), \
        "Output values differ between uncompiled and compiled versions"

if __name__ == "__main__":
    test_compile_jacfwd_one_hot_dynamic()
    print("Test passed successfully.")