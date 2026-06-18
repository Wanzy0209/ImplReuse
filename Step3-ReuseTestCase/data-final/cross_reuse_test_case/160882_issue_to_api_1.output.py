import torch
import pytest

def test_torch_compile_complex_dynamic_shapes():
    """
    Test case for Issue 160882: torch.compile crashes when torch.ops.aten.complex.default
    gets different inputs than when it was compiled.
    
    This test preserves the original bug reproduction logic, defining a function
    that creates a complex tensor and performs an IRFFT, then calling it with
    different input shapes after compilation.
    """
    
    # Define the function to be compiled, mirroring the structure of the
    # similar API's processing function (taking inputs and returning a result).
    def process_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        z = torch.complex(real, imag)
        return torch.fft.irfft(z, dim=1)

    B, F, T = 1, 641, 39

    # Create source tensors
    r_src = torch.randn(B, F, T)
    i_src = torch.randn(B, F, T)
    
    # Create mismatched tensors (permuted dimensions)
    r_mismatch = r_src.permute(0, 2, 1)
    i_mismatch = i_src.permute(0, 2, 1)

    # Compile the function with fullgraph and dynamic=True to handle shape changes
    # Note: The bug report shows fullgraph=True, and the traceback implies dynamic=True
    # was used to attempt to handle the shape mismatch.
    compiled_fn = torch.compile(process_complex, fullgraph=True, dynamic=True)

    # First call with original shape (1, 641, 39)
    # This establishes the initial compilation/trace
    try:
        out1 = compiled_fn(r_src, i_src)
        assert out1 is not None
        assert out1.shape == (B, 2 * (F - 1), T)
    except Exception as e:
        pytest.fail(f"First call failed: {e}")

    # Second call with mismatched shape (1, 39, 641)
    # This is the scenario that triggered the AssertionError in the bug report.
    try:
        out2 = compiled_fn(r_mismatch, i_mismatch)
        assert out2 is not None
        assert out2.shape == (B, 2 * (T - 1), F)
    except AssertionError as e:
        # Re-raise if it's the specific bug, though ideally the fix prevents this.
        pytest.fail(f"Compilation failed on dynamic shape change (Bug 160882): {e}")
    except Exception as e:
        pytest.fail(f"Second call failed with unexpected error: {e}")

if __name__ == "__main__":
    test_torch_compile_complex_dynamic_shapes()
    print("Test passed successfully.")