import torch

# Test case for torch.outer on MPS backend, adapted from the torch.clamp bug report.
# The original bug involved incorrect handling of zeros and small floats (1e-7).
# This test verifies that torch.outer handles these boundary conditions correctly on MPS.

def test_outer_mps():
    # Test 1: Outer product of zeros and 1e-7
    # Analogous to clamping zeros to a min of 1e-7
    a = torch.zeros(1, device='mps')
    b = torch.tensor([1e-7], device='mps')
    c = torch.outer(a, b)
    expected = torch.tensor([[0.]], device='mps')
    assert torch.allclose(c, expected), f"Test 1 Failed: Expected {expected}, got {c}"
    print(f"Test 1 Passed: {c}")

    # Test 2: Outer product of 1e-7 and zeros
    a = torch.tensor([1e-7], device='mps')
    b = torch.zeros(1, device='mps')
    c = torch.outer(a, b)
    expected = torch.tensor([[0.]], device='mps')
    assert torch.allclose(c, expected), f"Test 2 Failed: Expected {expected}, got {c}"
    print(f"Test 2 Passed: {c}")

    # Test 3: Outer product of 1e-7 and 1e-7
    # Checks precision handling with small floats
    a = torch.tensor([1e-7], device='mps')
    b = torch.tensor([1e-7], device='mps')
    c = torch.outer(a, b)
    expected = torch.tensor([[1e-14]], device='mps')
    assert torch.allclose(c, expected), f"Test 3 Failed: Expected {expected}, got {c}"
    print(f"Test 3 Passed: {c}")

    # Test 4: Mixed values including infinity
    # Analogous to clamp(min=1e-7, max=torch.inf)
    a = torch.tensor([0.0, 1e-7], device='mps')
    b = torch.tensor([1e-7, torch.inf], device='mps')
    c = torch.outer(a, b)
    # 0 * 1e-7 = 0
    # 0 * inf = nan
    # 1e-7 * 1e-7 = 1e-14
    # 1e-7 * inf = inf
    expected = torch.tensor([[0.0, float('nan')], [1e-14, float('inf')]], device='mps')
    # Use isnan for the nan check
    assert torch.allclose(c[0, 0], expected[0, 0]), f"Test 4a Failed"
    assert torch.isnan(c[0, 1]) and torch.isnan(expected[0, 1]), f"Test 4b Failed"
    assert torch.allclose(c[1, 0], expected[1, 0]), f"Test 4c Failed"
    assert torch.isinf(c[1, 1]) and torch.isinf(expected[1, 1]), f"Test 4d Failed"
    print(f"Test 4 Passed: {c}")

if __name__ == "__main__":
    test_outer_mps()