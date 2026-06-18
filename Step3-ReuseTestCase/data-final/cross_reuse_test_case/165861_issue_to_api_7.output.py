import torch
import torch.special

def test_special_erf_large_dimensions():
    """
    Test torch.special.erf with tensor dimensions exceeding uint16 max (2**16).
    This test is derived from the issue where F.pad (reflect mode) failed on CUDA
    when a dimension size was >= 2**16. We verify if torch.special.erf handles
    these large dimension sizes correctly.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Case 1: First dimension is exactly 2**16 (The breaking case in the original issue)
    x = torch.rand(2**16, 2, device="cuda")
    y = torch.special.erf(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"Test passed for shape {x.shape}")

    # Case 2: Middle dimension is 2**16
    x = torch.rand(1, 2**16, 2, device="cuda")
    y = torch.special.erf(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"Test passed for shape {x.shape}")

    # Case 3: Control case (2**16 - 1) - Should work fine
    x = torch.rand(2**16 - 1, 200, device="cuda")
    y = torch.special.erf(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"Test passed for shape {x.shape}")

    # Case 4: Large dimension in a different position (2**18)
    # The original issue noted this was fine for F.pad, but we verify for erf.
    x = torch.rand(2, 2**18, device="cuda")
    y = torch.special.erf(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"Test passed for shape {x.shape}")

if __name__ == "__main__":
    test_special_erf_large_dimensions()