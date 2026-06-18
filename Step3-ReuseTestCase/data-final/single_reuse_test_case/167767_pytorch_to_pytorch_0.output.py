import torch

def test_clamp_mps_min_value():
    """
    Test case for Issue 167767: clamp incorrectness with mps backend.
    Verifies that torch.clamp correctly applies the minimum value on MPS tensors.
    """
    if not torch.backends.mps.is_available():
        print("MPS backend not available, skipping test.")
        return

    # The trigger sequence described in the bug report.
    # The bug report indicates that calling clamp(min=0.0) on a previous tensor
    # triggers incorrect behavior for subsequent clamp operations on the MPS backend.
    a = torch.zeros(1, device='mps')
    a.clamp(min=0.0)

    # Test 1: Basic clamp with min
    b = torch.zeros(1, device='mps')
    c = b.clamp(min=1e-7)
    # Expected: 1e-7, Bug behavior: 0.0
    assert torch.isclose(c, torch.tensor([1e-7], device='mps')), \
        f"clamp(min=1e-7) failed: expected 1e-7, got {c.item()}"

    # Test 2: Clamp with min and max=None
    b = torch.zeros(1, device='mps')
    c = b.clamp(min=1e-7, max=None)
    assert torch.isclose(c, torch.tensor([1e-7], device='mps')), \
        f"clamp(min=1e-7, max=None) failed: expected 1e-7, got {c.item()}"

    # Test 3: Clamp with min and max=inf
    b = torch.zeros(1, device='mps')
    c = b.clamp(min=1e-7, max=torch.inf)
    assert torch.isclose(c, torch.tensor([1e-7], device='mps')), \
        f"clamp(min=1e-7, max=inf) failed: expected 1e-7, got {c.item()}"

    # Test 4: clamp_min
    b = torch.zeros(1, device='mps')
    c = b.clamp_min(1e-7)
    assert torch.isclose(c, torch.tensor([1e-7], device='mps')), \
        f"clamp_min(1e-7) failed: expected 1e-7, got {c.item()}"

    print("All clamp tests passed.")

if __name__ == "__main__":
    test_clamp_mps_min_value()