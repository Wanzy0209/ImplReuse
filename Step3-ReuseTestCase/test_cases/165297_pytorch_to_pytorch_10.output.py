import torch
import torch.nn as nn

def test_bceloss_channels_last_bfloat16():
    """
    Test case adapted from Issue 165297.
    Verifies if torch.nn.BCELoss produces NaNs or errors with large tensors,
    bfloat16 dtype, and channels_last memory format on CUDA.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    device = torch.device("cuda")

    # Large input tensor dimensions from the original bug report
    N, C, H, W = 84, 64, 512, 960

    # Case 1: bfloat16 + channels_last
    # BCELoss expects inputs to be probabilities (0-1).
    # We generate random data and pass through sigmoid to ensure valid probabilities.
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=device)
    x = torch.sigmoid(x)

    # Target needs to be same shape, values between 0 and 1.
    # We generate random 0s and 1s.
    target = torch.randint(0, 2, (N, C, H, W), dtype=torch.bfloat16, device=device)

    # Convert to NHWC channels_last layout
    x = x.to(memory_format=torch.channels_last)
    target = target.to(memory_format=torch.channels_last)

    # Uncommenting the lines below avoids NaNs in the original MaxPool bug
    # x = x.contiguous()
    # target = target.contiguous()

    print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Input stride: {x.stride()}")

    criterion = nn.BCELoss().to(device)
    
    try:
        loss = criterion(x, target)
        
        print(f"Output contains NaN? {torch.isnan(loss).any().item()}")
        print(f"Output contains Inf? {torch.isinf(loss).any().item()}")
        print(f"Stats: loss={loss.item()}")

        if torch.isnan(loss).any():
            print("Detected NaNs in BCELoss output!")
        else:
            print("Test passed: No NaNs detected.")
            
    except RuntimeError as e:
        print(f"RuntimeError encountered: {e}")

if __name__ == "__main__":
    test_bceloss_channels_last_bfloat16()