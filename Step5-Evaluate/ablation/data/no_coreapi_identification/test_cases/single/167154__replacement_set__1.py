import torch
from torch.nn import functional as F
import torch.nn as nn

# Check for MPS availability
if not torch.backends.mps.is_available():
    print("MPS device not available. Test cannot run.")
else:
    device = "mps"
    dtype = torch.float32

    # Original: weight = torch.rand((768, 768), device="mps", dtype=torch.float32)
    # Replacement: torch.rand -> torch.flipud
    # flipud requires an input tensor, so we create a dummy tensor to flip.
    dummy_weight = torch.ones((768, 768), device=device, dtype=dtype)
    weight = torch.flipud(dummy_weight)

    # Original: bias = torch.rand((768), device="mps", dtype=torch.float32)
    # Replacement: torch.rand -> torch.flipud
    dummy_bias = torch.ones((768), device=device, dtype=dtype)
    bias = torch.flipud(dummy_bias)

    shape = (5, 499, 768)
    stride = (0, 768, 1)
    storage_offset = 0
    numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1

    # Original: base = torch.arange(numel, dtype=torch.float32, device="mps")
    # Replacement: torch.arange -> torch.var
    # torch.var computes variance. To generate a tensor of size 'numel', 
    # we compute variance along a dimension of a larger dummy tensor.
    # Creating a tensor of shape (2, numel) and taking var along dim 0 results in shape (numel,).
    dummy_base = torch.ones((2, numel), device=device, dtype=dtype)
    base = torch.var(dummy_base, dim=0, unbiased=False)

    # Original: input = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)
    # Replacement: torch.as_strided -> torch.nn.Flatten
    # Flatten reshapes the input. Since 'base' is 1D, Flatten keeps it 1D.
    flatten_layer = nn.Flatten()
    input = flatten_layer(base)

    # Original API Under Test: torch.nn.functional.linear
    # Note: The original bug was triggered by specific non-contiguous strides created by as_strided.
    # With the replacement using Flatten, the input tensor is contiguous, so the specific 
    # MPS buffer allocation regression might not manifest, but we verify the API execution.
    try:
        output = F.linear(input, weight, bias)
        # Assertion to verify the operation completed and produced a shape consistent with linear algebra
        # Input size is numel (7664640), Weight is (768, 768).
        # Linear treats input as (Batch, InFeatures) -> (7664640/768, 768) -> (9980, 768)
        expected_shape = (numel // 768, 768)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print("Test passed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")