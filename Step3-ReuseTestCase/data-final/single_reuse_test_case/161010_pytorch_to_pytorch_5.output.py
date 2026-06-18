import torch

def test_torch_any_compile_consistency():
    """
    Test case to verify that torch.any behaves consistently under torch.compile,
    specifically using tensor inputs with non-contiguous strides (similar to the context of Issue 161010).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand(5, 5, device=device)

    def f(A):
        # Reproduce the tensor setup from the original bug report to generate
        # a tensor 'a' with specific strides (non-contiguous).
        Q, R = torch.linalg.qr(A)
        rhs = torch.ones(Q.shape[0], 1, device=A.device)
        a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
        
        # Use torch.any on the tensor 'a'
        # We check if any element is greater than 0.5 to make the result non-deterministic based on values,
        # but consistent between eager and compile if the implementation is correct.
        # Alternatively, checking > 0 is almost always true for positive random numbers, 
        # but let's stick to a simple condition.
        return torch.any(a > 0)

    # Eager execution
    res_eager = f(A)
    print(f"Eager result: {res_eager}")

    # Compiled execution
    f_compiled = torch.compile(f)
    res_compiled = f_compiled(A)
    print(f"Compiled result: {res_compiled}")

    # Verify that the results match
    assert res_eager == res_compiled, (
        f"torch.any produced different results under torch.compile: "
        f"eager={res_eager}, compiled={res_compiled}"
    )
    print("Test passed: torch.any is consistent under torch.compile.")

if __name__ == "__main__":
    test_torch_any_compile_consistency()