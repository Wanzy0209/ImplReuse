import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    # Adapt the failing case for conv2d.
    # conv2d expects 4D input (Batch, Channels, Height, Width).
    # We set the Batch dimension to 2**16 (65536), which is > uint16 max (65535).
    x = torch.rand(2**16, 2, 3, 3, device="cuda")
    
    # Define a simple weight tensor for convolution
    weight = torch.rand(1, 2, 3, 3, device="cuda")

    # Test 'reflect' padding mode (suspected to fail based on the similar bug in F.pad)
    # Since F.conv2d in this environment does not support the 'padding_mode' argument directly,
    # we manually apply padding using F.pad and then call conv2d with padding=0.
    try:
        # padding=1 in conv2d corresponds to padding (left, right, top, bottom) = (1, 1, 1, 1)
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        out_reflect = F.conv2d(x_padded, weight, padding=0)
        print("conv2d reflect: OK")
    except RuntimeError as e:
        print(f"conv2d reflect: FAILED - {e}")

    # Test 'zeros' padding mode (expected to pass)
    try:
        x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        out_zeros = F.conv2d(x_padded, weight, padding=0)
        print("conv2d zeros: OK")
    except RuntimeError as e:
        print(f"conv2d zeros: FAILED - {e}")

    # Test 'replicate' padding mode (expected to pass)
    try:
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        out_replicate = F.conv2d(x_padded, weight, padding=0)
        print("conv2d replicate: OK")
    except RuntimeError as e:
        print(f"conv2d replicate: FAILED - {e}")
else:
    print("CUDA is not available. This test requires a CUDA device.")