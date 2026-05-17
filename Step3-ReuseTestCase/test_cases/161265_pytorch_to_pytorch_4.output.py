import torch

# Check if MPS is available to run the test
if torch.backends.mps.is_available():
    # Create a tensor larger than 4GB.
    # Shape: (2, 1<<31 + 5) -> 2 * (2,147,483,648 + 5) elements
    # dtype: int8 (1 byte per element)
    # Total size: ~4.29 GB + 10 bytes
    # Note: torch.ones uses torch.full internally, which is the API subject to the bug.
    a = torch.ones(2, (1 << 31) + 5, dtype=torch.int8, device='mps')

    # Test torch.amin
    # Since the tensor is filled with 1s, the minimum value should be 1.
    # If the bug (fillBuffer failure for >4GB) occurs, parts of the tensor will remain 0,
    # causing torch.amin to return 0.
    min_val = torch.amin(a)

    print(f"Minimum value: {min_val}")
    assert min_val.item() == 1, f"Expected minimum value 1, but got {min_val.item()}"
else:
    print("MPS device not found. Skipping test.")