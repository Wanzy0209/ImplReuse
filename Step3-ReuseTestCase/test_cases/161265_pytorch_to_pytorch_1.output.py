import torch

# Check if MPS is available
if not torch.backends.mps.is_available():
    print("MPS device not found. Skipping test.")
else:
    # Adapted test for torch.fmax
    # The original bug involved tensors larger than 4GB.
    # Original: int8, shape (2, (1 << 31) + 5) -> ~4GB
    # Adapted: float32 (4 bytes), shape (2, (1 << 29) + 5) -> ~4.3GB
    # This ensures we exceed the 4GB buffer limit that caused the bug.
    shape = (2, (1 << 29) + 5)
    
    a = torch.ones(shape, dtype=torch.float32, device='mps')
    b = torch.zeros(shape, dtype=torch.float32, device='mps')

    # Call the similar API: torch.fmax
    # fmax(a, b) should return a where a >= b
    c = torch.fmax(a, b)

    # Verify the result at the tail indices (the region affected by the fillBuffer bug)
    print(c[1, -2])
    print(c[:, -2])

    # Assertions to ensure correctness
    # Expected value is 1.0 because a is filled with 1.0 and b with 0.0
    assert c[1, -2] == 1.0, f"Expected 1.0 at [1, -2], got {c[1, -2]}"
    assert torch.all(c[:, -2] == 1.0), f"Expected 1.0 at [:, -2], got {c[:, -2]}"