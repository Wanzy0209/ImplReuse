import torch
import torch.nn as nn

def test_cosine_similarity_channels_last_bfloat16():
    """
    Test case adapted from Issue 165297 (MaxPool2d NaNs).
    
    This test verifies if torch.nn.CosineSimilarity exhibits similar numerical 
    instability (NaNs/Infs) or crashes when processing large tensors with 
    bfloat16 dtype and channels_last memory format on CUDA.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    device = torch.device("cuda")
    
    # Clear cache to ensure maximum available memory
    torch.cuda.empty_cache()

    # Reproduce the large tensor scenario from the MaxPool2d bug report
    # Fix: Reduced batch size N from 84 to 4 to prevent OutOfMemoryError 
    # while keeping H and W large to test channels_last behavior.
    N, C, H, W = 4, 64, 512, 960

    # Case 1: bfloat16 + channels_last (The problematic configuration in the bug)
    x1 = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=device)
    x2 = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=device)

    # Convert to NHWC channels_last layout
    x1 = x1.to(memory_format=torch.channels_last)
    x2 = x2.to(memory_format=torch.channels_last)

    # Verify memory format setup
    print(f"Input x1: contiguous={x1.is_contiguous()}, channels_last={x1.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Input x1 stride: {x1.stride()}")

    # Initialize CosineSimilarity
    # Using dim=1 (channel dimension) to compare feature vectors
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    # Run operation
    # Fix: Use torch.no_grad() to reduce memory usage during inference
    try:
        with torch.no_grad():
            y = cos_sim(x1, x2)
    except RuntimeError as e:
        print(f"RuntimeError encountered (e.g., illegal memory access): {e}")
        raise

    # Check for NaNs and Infs (Symptoms of the original bug)
    has_nan = torch.isnan(y).any().item()
    has_inf = torch.isinf(y).any().item()

    print(f"Output contains NaN? {has_nan}")
    print(f"Output contains Inf? {has_inf}")
    print(f"Stats: min={y.min().item()}, max={y.max().item()}")

    assert not has_nan, "Detected NaNs in CosineSimilarity output with channels_last + bfloat16"
    assert not has_inf, "Detected Infs in CosineSimilarity output with channels_last + bfloat16"

if __name__ == "__main__":
    test_cosine_similarity_channels_last_bfloat16()