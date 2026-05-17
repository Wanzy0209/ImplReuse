import torch

def test_compile_preserves_stride_with_clone():
    """
    Test case for Issue 161010.
    Verifies that torch.compile preserves stride information when using
    clone(memory_format=torch.preserve_format), matching eager mode behavior.
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand(5, 5, device=device)

    # The function containing the logic to be tested
    # This mirrors the structure of the similar API's function definition
    def check_stride_preservation(A, count):
        Q, R = torch.linalg.qr(A)
        rhs = torch.ones(Q.shape[0], 1, device=A.device)
        a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
        
        # The core check: does clone with preserve_format maintain the stride?
        if a.stride() == a.clone(memory_format=torch.preserve_format).stride():
            return count + 1
        return count

    # Execute in eager mode
    res_eager = check_stride_preservation(A, torch.zeros(1))

    # Execute in compiled mode
    compiled_func = torch.compile(check_stride_preservation)
    res_compiled = compiled_func(A, torch.zeros(1))

    # Assertion to catch the silent wrong result
    assert torch.equal(res_eager, res_compiled), (
        f"torch.compile failed to preserve stride with clone(memory_format=torch.preserve_format).\n"
        f"Eager result: {res_eager.item()}, Compiled result: {res_compiled.item()}"
    )

if __name__ == "__main__":
    test_compile_preserves_stride_with_clone()
    print("Test passed successfully.")