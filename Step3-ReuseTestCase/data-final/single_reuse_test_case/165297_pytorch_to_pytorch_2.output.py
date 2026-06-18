import torch
import torch.nn as nn

# Check for CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
else:
    device = torch.device("cuda")

    def test_avgpool3d_nan():
        # Large input tensor adapted for 3D pooling
        # Original: N, C, H, W = 84, 64, 512, 960
        # Adapted: N, C, D, H, W = 16, 64, 32, 512, 480
        N, C, D, H, W = 16, 64, 32, 512, 480

        # Case 1: bfloat16 + channels_last_3d
        x = torch.randn(N, C, D, H, W, dtype=torch.bfloat16, device=device)

        # Convert to channels_last_3d layout (NCDHW -> NDHWC memory format)
        # Note: For 5D tensors, we use channels_last_3d instead of channels_last
        x = x.to(memory_format=torch.channels_last_3d)

        # Uncommenting the line below avoids NaNs (if the bug exists)
        # x = x.contiguous()

        print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last_3d={x.is_contiguous(memory_format=torch.channels_last_3d)}")
        print(f"Input stride: {x.stride()}")

        # AvgPool3d with 3D parameters
        pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1).to(device)
        y = pool(x)

        print(f"Output contains NaN? {torch.isnan(y).any().item()}")
        print(f"Output contains Inf? {torch.isinf(y).any().item()}")
        print(f"Stats: min={y.min().item()}, max={y.max().item()}")

        if torch.isnan(y).any():
            print("Detected NaNs in AvgPool3d output!")

    test_avgpool3d_nan()