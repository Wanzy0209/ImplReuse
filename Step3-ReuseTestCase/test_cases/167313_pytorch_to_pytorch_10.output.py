import torch
import torch.nn.functional as F

def test_fmax_compilation():
    """
    Test case to verify torch.fmax behavior under torch.compile.
    Adapted from the addmm bug report structure to ensure decomposition
    correctness and parameter handling (though fmax has no scalar params,
    we verify the core logic and NaN handling).
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Setup inputs on CUDA
    # Include NaNs to verify the specific fmax logic (NaN propagation)
    x = torch.tensor([1.0, float('nan'), 5.0], device="cuda")
    y = torch.tensor([2.0, 3.0, float('nan')], device="cuda")

    # Define the function using torch.fmax
    # We wrap it in a point-wise operation (relu) similar to the original bug report
    # to ensure the decomposition path is exercised.
    f = lambda x, y: F.relu(torch.fmax(x, y))

    # Compile the function
    fc = torch.compile(f)

    # Execute eager and compiled versions
    res_eager = f(x, y)
    res_compiled = fc(x, y)

    print("Eager result:  ", res_eager)
    print("Compiled result:", res_compiled)

    # Assert that the results are identical, including NaN handling
    assert torch.allclose(res_eager, res_compiled, equal_nan=True), \
        f"Mismatch between eager and compiled results:\nEager: {res_eager}\nCompiled: {res_compiled}"

    print("Test passed.")

if __name__ == "__main__":
    test_fmax_compilation()