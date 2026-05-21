import torch
import torch.nn.functional as F

def test_sigmoid_mps_correctness():
    """
    Test case for torch.nn.functional.sigmoid on the MPS backend.
    This test mirrors the logic of the clamp bug report (Issue 167767),
    checking for incorrect behavior on the MPS device, specifically
    handling of zero inputs and expression patterns found in similar API usage.
    """
    if not torch.backends.mps.is_available():
        print("MPS backend is not available. Skipping test.")
        return

    print("Testing torch.nn.functional.sigmoid on MPS backend...")

    # Test 1: Basic functional call with zero input
    # Mirrors: b = torch.zeros(1, device='mps'); c = b.clamp(min=1e-7)
    a = torch.zeros(1, device='mps')
    print(f"Input: {a}")
    c = F.sigmoid(a)
    print(f"Output: {c}")
    # Sigmoid(0) should be 0.5. If MPS backend is broken like clamp was, this might fail.
    assert torch.allclose(c, torch.tensor([0.5], device='mps')), \
        f"Expected 0.5, got {c.item()}"

    # Test 2: Method call with zero input
    # Mirrors: c = b.clamp_min(1e-7)
    b = torch.zeros(1, device='mps')
    print(f"Input: {b}")
    c = b.sigmoid()
    print(f"Output: {c}")
    assert torch.allclose(c, torch.tensor([0.5], device='mps')), \
        f"Expected 0.5, got {c.item()}"

    # Test 3: Expression pattern based on Similar API information
    # The similar API info showed usage like: return (3 * a).sigmoid()
    # We test this pattern to ensure the backend handles chained operations correctly.
    d = torch.tensor([1.0], device='mps')
    print(f"Input: {d}")
    c = (3 * d).sigmoid()
    print(f"Output (3*d).sigmoid(): {c}")
    # 3 * 1.0 = 3.0, sigmoid(3.0) approx 0.9525741268224334
    expected_val = torch.sigmoid(torch.tensor([3.0]))
    assert torch.allclose(c, expected_val), \
        f"Expected {expected_val.item()}, got {c.item()}"

    print("All sigmoid tests passed on MPS backend.")

if __name__ == "__main__":
    test_sigmoid_mps_correctness()