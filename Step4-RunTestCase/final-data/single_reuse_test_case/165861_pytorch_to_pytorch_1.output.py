import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    # Create input tensor with batch dimension = 2**16
    # Shape: (Batch, Channels, Length)
    x = torch.rand(2**16, 2, 10, device="cuda")

    # Create weight tensor
    # Shape: (Out_Channels, In_Channels, Kernel_Size)
    weight = torch.rand(2, 2, 3, device="cuda")

    # Test with reflect padding
    # The error indicates that padding_mode is not supported directly in F.conv1d.
    # We manually apply reflect padding using F.pad, then run conv1d with padding=0.
    try:
        # F.pad format for 1D is (left, right)
        x_padded = F.pad(x, pad=(1, 1), mode='reflect')
        out_reflect = F.conv1d(x_padded, weight, padding=0)
        print("conv1d reflect padding ok")
    except RuntimeError as e:
        print(f"conv1d reflect padding failed: {e}")

    # Test with zeros padding
    # padding_mode='zeros' is the default behavior, so we simply remove the argument.
    try:
        out_zeros = F.conv1d(x, weight, padding=1)
        print("conv1d zeros padding ok")
    except RuntimeError as e:
        print(f"conv1d zeros padding failed: {e}")
else:
    print("CUDA is not available. This test requires a CUDA device.")