import torch
import torch.special
import os

# Reproducer for Issue 164700 adapted for torch.special.psi
# The original issue involves torch.compile crashing with 'constexpr_type' object has no attribute 'is_block'
# when handling interleaved unsqueeze and torch.cat operations.
# This test case verifies that torch.compile handles these patterns correctly when 
# torch.special.psi is introduced into the computation graph.

def test_compile_psi_with_cat_unsqueeze():
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch 2.0+). Skipping test.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # torch.special.psi requires floating point inputs (unlike the original int64/int32 repro)
    # We use positive values to avoid singularities at 0 and negative integers for the digamma function
    x = torch.ones(1, 32, dtype=torch.float32, device=device)
    y = torch.ones(1, dtype=torch.float32, device=device)

    def f(x, y):
        # Integrate torch.special.psi into the logic
        y_psi = torch.special.psi(y)
        
        # Original problematic pattern: cat with unsqueeze
        y2 = torch.cat(
            [
                x[:, 1:],
                y_psi[:, None] + 32 * 2048,
            ],
            dim=1,
        )

        x2 = x[:, 1:, None]
        y3 = y2[:, -1:, None]

        # Use torch.special.psi on the broadcasted range
        # Shifted range to 1..2048 to ensure valid inputs for psi
        range_vals = torch.arange(1, 2049, device=device, dtype=torch.float32)
        
        return (
            torch.cat([x2, y3], dim=1)
            + torch.special.psi(range_vals)[None, None, :]
        ).reshape(1, 32 * 2048)

    # Run eager mode to establish baseline
    try:
        expected = f(x, y)
    except Exception as e:
        print(f"Eager execution failed: {e}")
        return

    # Run compiled mode
    # This is where the original bug (constexpr_type crash) would manifest
    # if the compiler backend issues persist with this API usage.
    compiled_f = torch.compile(f)
    try:
        result = compiled_f(x, y)
    except Exception as e:
        print(f"Compiled execution failed: {e}")
        raise

    # Verify correctness
    assert torch.allclose(expected, result), "Compiled output differs from eager output"

if __name__ == "__main__":
    test_compile_psi_with_cat_unsqueeze()