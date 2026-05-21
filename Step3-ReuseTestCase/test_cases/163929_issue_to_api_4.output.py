import torch

def test_inductor_reduction_on_transposed_mutated_matrix():
    """
    Test case for Issue 163929.
    
    Verifies that argmin and argmax produce correct results when operating on
    a matrix that has been mutated in-place and then transposed.
    
    This test leverages the pattern found in tf.compat.v1.distributions.Categorical,
    which relies on argmax logic for sampling, to ensure the inductor backend
    handles reductions correctly after in-place mutations and view operations.
    """
    
    # Original function from the bug report using argmin
    def func_argmin(x):
        x.tan_()
        x = x.t()
        return x.argmin()

    # Related function using argmax, aligning with the Similar API (Categorical)
    # and the Issue Title.
    def func_argmax(x):
        x.tan_()
        x = x.t()
        return x.argmax()

    torch.manual_seed(0)

    # Test Argmin
    x1 = torch.randn(4, 6)
    x2 = x1.clone()
    
    out_eager_min = func_argmin(x1)
    compiled_min = torch.compile(func_argmin)
    out_compiled_min = compiled_min(x2)
    
    torch.testing.assert_close(out_eager_min, out_compiled_min)

    # Test Argmax
    x3 = torch.randn(4, 6)
    x4 = x3.clone()
    
    out_eager_max = func_argmax(x3)
    compiled_max = torch.compile(func_argmax)
    out_compiled_max = compiled_max(x4)
    
    torch.testing.assert_close(out_eager_max, out_compiled_max)

if __name__ == "__main__":
    test_inductor_reduction_on_transposed_mutated_matrix()