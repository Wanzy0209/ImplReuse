import torch
import torch.testing

def test_compile_incorrect_fusion():
    """
    Test case for Issue 162410: incorrect fusion cause numerical issue.
    Verifies that torch.compile handles in-place copy_ with view operations (flip)
    and reductions correctly without numerical discrepancies.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    def f(x, y):
        x.copy_(x.flip(1))
        y = y.sum(dim=1, keepdim=True) + y
        return x + y

    # Initialize inputs on CUDA
    x = torch.randn(20, 1024 * 1024, device="cuda")
    x_copy = x.clone()
    y = torch.randn(20, 1024 * 1024, device="cuda")

    # Compile the function using torch.compile
    opt_f = torch.compile(f)

    # Execute eager and compiled versions
    ref = f(x, y)
    act = opt_f(x_copy, y)

    # Assert that the results are numerically close
    torch.testing.assert_close(ref, act)
    print("Test passed.")

if __name__ == "__main__":
    test_compile_incorrect_fusion()