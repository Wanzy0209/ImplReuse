import torch
import torch.special

def test_special_erfc_large_dimension():
    """
    Test case adapted from Issue 165861.
    
    The original issue reported that torch.nn.functional.pad with mode='reflect' 
    crashed on CUDA when a batch dimension was >= 2**16 (uint16 max).
    Given the high code similarity, this test verifies if torch.special.erfc
    handles large dimension sizes correctly without crashing or producing invalid results.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Case 1: First dimension is exactly 2**16 (The breaking case in the original issue)
    x = torch.rand(2**16, 2, device="cuda")
    y = torch.special.erfc(x)
    assert y.shape == x.shape, "Output shape mismatch"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    print("Case 1 (dim 0 = 2**16) passed")

    # Case 2: Middle dimension is 2**16
    x = torch.rand(1, 2**16, 2, device="cuda")
    y = torch.special.erfc(x)
    assert y.shape == x.shape, "Output shape mismatch"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    print("Case 2 (dim 1 = 2**16) passed")

    # Case 3: Last dimension is 2**16
    x = torch.rand(2, 2**16, device="cuda")
    y = torch.special.erfc(x)
    assert y.shape == x.shape, "Output shape mismatch"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    print("Case 3 (dim 1 = 2**16) passed")

    print("All tests passed for torch.special.erfc with large dimensions.")

if __name__ == "__main__":
    test_special_erfc_large_dimension()