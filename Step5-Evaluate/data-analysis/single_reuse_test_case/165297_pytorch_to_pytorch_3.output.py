import torch
import torch.nn as nn

# Ensure CUDA is available for the test
if not torch.cuda.is_available():
    print("Test skipped: CUDA is not available.")
else:
    device = torch.device("cuda")

    def test_maxpool3d_nan():
        # Reduced dimensions to fit within GPU memory constraints
        # Original shape (84, 64, 16, 512, 960) required ~80GB memory
        # New shape (4, 64, 16, 128, 128) requires ~134MB memory
        N, C, D, H, W = 4, 64, 16, 128, 128
        
        # Case 1: bfloat16 + channels_last (NDHWC) => Potential NaN
        x = torch.randn(N, C, D, H, W, dtype=torch.bfloat16, device=device)

        # Convert to channels_last layout (NDHWC for 5D tensors)
        x = x.to(memory_format=torch.channels_last)

        # Uncommenting the line below avoids NaNs in the original bug
        # x = x.contiguous()

        print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
        print(f"Input stride: {x.stride()}")

        # Use MaxPool3d instead of MaxPool2d
        pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1).to(device)
        y = pool(x)

        print(f"Output contains NaN? {torch.isnan(y).any().item()}")
        print(f"Output contains Inf? {torch.isinf(y).any().item()}")
        print(f"Stats: min={y.min().item()}, max={y.max().item()}")

        if torch.isnan(y).any():
            print("Detected NaNs in MaxPool3d output!")
        else:
            print("No NaNs detected.")

    test_maxpool3d_nan()