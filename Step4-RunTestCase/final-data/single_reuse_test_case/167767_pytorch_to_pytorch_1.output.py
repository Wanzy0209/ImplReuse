import torch

def test_clip_mps_incorrectness():
    if not torch.backends.mps.is_available():
        print("MPS backend is not available. Skipping test.")
        return

    print("Testing torch.clip on MPS backend...")

    # The following line triggers the incorrect behavior in the original bug report
    a = torch.zeros(1, device='mps')
    a_clipped = a.clip(min=0.0)

    # Test 1: clip with min argument
    b = torch.zeros(1, device='mps')
    c = b.clip(min=1e-7)
    print(f"Test 1 - clip(min=1e-7): {c}")
    # Expected: tensor([1e-07], device='mps:0')
    # Buggy behavior: tensor([0.], device='mps:0')
    assert torch.allclose(c, torch.tensor([1e-7], device='mps')), \
        f"Expected 1e-7, but got {c.item()}"

    # Test 2: clip with min and max=None
    b = torch.zeros(1, device='mps')
    c = b.clip(min=1e-7, max=None)
    print(f"Test 2 - clip(min=1e-7, max=None): {c}")
    assert torch.allclose(c, torch.tensor([1e-7], device='mps')), \
        f"Expected 1e-7, but got {c.item()}"

    # Test 3: clip with min and max=torch.inf
    b = torch.zeros(1, device='mps')
    c = b.clip(min=1e-7, max=torch.inf)
    print(f"Test 3 - clip(min=1e-7, max=inf): {c}")
    assert torch.allclose(c, torch.tensor([1e-7], device='mps')), \
        f"Expected 1e-7, but got {c.item()}"

    print("All tests passed.")

if __name__ == "__main__":
    test_clip_mps_incorrectness()