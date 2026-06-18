import torch

if torch.backends.mps.is_available():
    # Create a tensor larger than 4GB using torch.full
    # Size: 2 * (2^31 + 5) elements
    # Dtype: int8 (1 byte per element)
    # Total size: ~4GB + 10 bytes
    size = (2, (1 << 31) + 5)
    fill_value = 1
    
    a = torch.full(size, fill_value, dtype=torch.int8, device='mps')

    # Check specific elements that were failing in the bug report
    # These elements are located beyond the 4GB offset in the buffer
    val_single = a[1, -2].item()
    val_slice = a[:, -2]

    print(f"Value at a[1, -2]: {val_single}")
    print(f"Value at a[:, -2]: {val_slice}")

    # Assertions to verify the fix
    assert val_single == fill_value, f"Expected {fill_value}, got {val_single}"
    assert torch.all(val_slice == fill_value), f"Expected all {fill_value}, got {val_slice}"
    
    print("Test passed.")
else:
    print("MPS device not found. Skipping test.")