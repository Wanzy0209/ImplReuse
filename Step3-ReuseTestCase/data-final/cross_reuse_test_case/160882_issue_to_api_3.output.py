import torch
import pytest

def test_torch_compile_complex_dynamic_shape_mismatch():
    """
    Test case for Issue 160882.
    
    This test reproduces the bug where torch.compile crashes with an AssertionError
    when torch.ops.aten.complex.default receives inputs with different shapes
    than those seen during the initial compilation pass.
    
    The test structure mirrors the pattern of the similar API (tf.keras.initializers.serialize)
    by defining a processing function and verifying its behavior under compilation,
    specifically focusing on the handling of input variations.
    """
    
    # Define the function containing the complex operation
    # This is the core logic being tested for compilation stability.
    def f(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        z = torch.complex(real, imag)
        return torch.fft.irfft(z, dim=1)

    # Compile the function with fullgraph=True to enforce strict compilation
    compiled = torch.compile(f, fullgraph=True)

    B, F, T = 1, 641, 39

    # Create source tensors with shape (1, 641, 39)
    r_src = torch.randn(B, F, T)
    i_src = torch.randn(B, F, T)

    # First call: compiles the graph based on this shape
    # This should succeed.
    try:
        result_1 = compiled(r_src, i_src)
        assert result_1 is not None
    except Exception as e:
        pytest.fail(f"Compilation or first execution failed: {e}")

    # Create mismatched tensors with shape (1, 39, 641) via permutation
    r_mismatch = r_src.permute(0, 2, 1)
    i_mismatch = i_src.permute(0, 2, 1)

    # Second call: attempts to reuse the compiled graph with a different shape.
    # In the bugged version, this raises an AssertionError.
    # We use pytest.raises to verify the bug exists (or remove it to verify the fix).
    with pytest.raises(AssertionError):
        _ = compiled(r_mismatch, i_mismatch)

if __name__ == "__main__":
    test_torch_compile_complex_dynamic_shape_mismatch()