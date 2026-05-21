import torch

# Adapted test case for torch.nn.MaxUnpool2d based on the MaxUnpool3d bug report.
# The original bug involved a segmentation fault, likely triggered by invalid tensor types
# (complex128, uint32) and mismatched shapes passed to the forward pass.
# To verify if MaxUnpool2d is susceptible to a similar crash, we adapt the tensor dimensions
# from 5D (for 3D) to 4D (for 2D) and provide a valid kernel_size to ensure initialization
# reaches the forward pass logic.

# Note: We adapt input[0] to include kernel_size=(2, 2) because MaxUnpool2d requires it.
# Without it, the test would fail at initialization with a TypeError, preventing us from
# checking for the segmentation fault in the forward pass.

input_data = [
    [(2, 2)],  # kernel_size (adapted from empty tuple to allow initialization)
    {},        # kwargs for init
    [
        # Adapted from 5D (9, 6, 3, 6, 9) to 4D (9, 6, 6, 9) for MaxUnpool2d
        torch.empty((9, 6, 6, 9), dtype=torch.complex128, device='cuda'),
        # Adapted from 5D (5, 7, 9, 8, 5) to 4D (5, 7, 8, 5) for MaxUnpool2d
        torch.empty((5, 7, 8, 5), dtype=torch.uint32, device='cuda')
    ],
    {}         # kwargs for forward
]

print(f"Testing torch.nn.MaxUnpool2d with complex128 input and uint32 indices on CUDA")
try:
    # Initialize MaxUnpool2d
    r1 = torch.nn.MaxUnpool2d(*input_data[0], **input_data[1])
    # Forward pass with potentially crashing inputs
    r2 = r1(*input_data[2], **input_data[3])
    print("Test passed: No crash occurred.")
except RuntimeError as e:
    # Expected behavior might be a runtime error regarding types or shapes,
    # but a segmentation fault indicates a bug.
    print(f"Runtime Error (Expected): {e}")
except Exception as e:
    print(f"Unexpected Exception: {e}")