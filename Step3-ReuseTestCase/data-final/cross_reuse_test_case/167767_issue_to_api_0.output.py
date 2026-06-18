import torch
import torch.nn.functional as F

def test_tanh_mps_correctness():
    """
    Test case for torch.nn.functional.tanh on the MPS backend.
    This test preserves the reproduction logic from the clamp bug report (Issue 167767),
    which involved verifying element-wise operations on specific values (0 and 1e-7)
    using the MPS device.
    """
    if not torch.backends.mps.is_available():
        print("MPS backend not available, skipping test.")
        return

    # Test 1: Reproduce the setup with zero input
    # The original bug report started with a = torch.zeros(1, device='mps')
    a = torch.zeros(1, device='mps')
    
    # Apply the similar API (tanh) instead of clamp
    c = F.tanh(a)
    
    # Verify the result. tanh(0) should be 0.
    # The original bug showed that clamp(min=1e-7) incorrectly returned 0.
    # Here we verify tanh returns the mathematically correct 0.
    expected = torch.tensor([0.0], device='mps')
    assert torch.allclose(c, expected), f"tanh(0) failed on MPS. Expected {expected}, got {c}"

    # Test 2: Reproduce the setup with the problematic small value (1e-7)
    # The original bug report tested clamping to 1e-7.
    b = torch.tensor([1e-7], device='mps')
    
    # Apply tanh
    c = F.tanh(b)
    
    # For small x, tanh(x) is approximately x.
    # We check if the backend handles this small value correctly.
    assert torch.allclose(c, b, atol=1e-9), f"tanh(1e-7) failed on MPS. Expected ~{b}, got {c}"

    # Test 3: Verify with a standard value to ensure general operation
    d = torch.ones(1, device='mps')
    c = F.tanh(d)
    expected_val = torch.tensor([torch.tanh(torch.tensor(1.0))], device='mps')
    assert torch.allclose(c, expected_val), f"tanh(1.0) failed on MPS. Expected {expected_val}, got {c}"

    print("All tanh tests passed on MPS.")

if __name__ == "__main__":
    test_tanh_mps_correctness()