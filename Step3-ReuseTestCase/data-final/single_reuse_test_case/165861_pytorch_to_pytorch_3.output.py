import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    # Adapt the test case for conv3d.
    # conv3d requires 5D input (Batch, Channel, Depth, Height, Width).
    # We test the scenario where the batch dimension is 2**16.
    batch_size = 2**16
    x = torch.rand(batch_size, 1, 3, 3, 3, device="cuda")
    
    # conv3d requires a weight tensor of shape (OutChannels, InChannels/Groups, kD, kH, kW)
    weight = torch.rand(1, 1, 3, 3, 3, device="cuda")

    # Test with reflect padding mode, which was the failing mode in F.pad
    try:
        F.conv3d(x, weight, padding=1, padding_mode="reflect")
        print("conv3d reflect padding ok")
    except RuntimeError as e:
        print(f"conv3d reflect padding failed: {e}")

    # Test with constant padding mode as a control
    try:
        F.conv3d(x, weight, padding=1, padding_mode="zeros")
        print("conv3d constant padding ok")
    except RuntimeError as e:
        print(f"conv3d constant padding failed: {e}")
else:
    print("CUDA not available, skipping test.")