# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn as nn

device = torch.device("cuda")

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    # Case 1: bfloat16 + channels_last  => NaN
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=device)

    # Case 2: float32 + channels_last => illegal memory access
    # x = torch.randn(N, C, H, W, device=device)  # default float32


    # Convert to NHWC channels_last layout
    x = x.to(memory_format=torch.channels_last)

    # Uncommenting the line below avoids NaNs
    # x = x.contiguous()

    print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Input stride: {x.stride()}")

    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
    y = pool(x)

    print(f"Output contains NaN? {torch.isnan(y).any().item()}")
    print(f"Output contains Inf? {torch.isinf(y).any().item()}")
    print(f"Stats: min={y.min().item()}, max={y.max().item()}")

    if torch.isnan(y).any():
        print("Detected NaNs in MaxPool output!")

test_maxpool_nan()