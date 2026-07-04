import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    # Adapt the test case for torch.nn.functional.grid_sample
    # The bug is related to large batch dimensions (> 2**16) with reflection padding.
    # grid_sample requires 4D (N, C, H, W) or 5D inputs.
    
    batch_size = 2**16
    # Create input tensor with large batch dimension
    input_tensor = torch.rand(batch_size, 2, 4, 4, device="cuda")
    
    # Create grid tensor matching the batch size
    # grid_sample expects normalized coordinates in [-1, 1]
    grid = torch.rand(batch_size, 4, 4, 2, device="cuda") * 2 - 1

    # Test with 'reflection' padding mode (analogous to 'reflect' in F.pad)
    # This is expected to fail or trigger the bug if the underlying issue exists in grid_sample
    try:
        output = F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='reflection')
        print("grid_sample with reflection padding ok")
    except RuntimeError as e:
        print(f"grid_sample failed with reflection padding: {e}")

    # Test with 'zeros' padding mode to verify other modes work (as per original bug report context)
    try:
        output = F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='zeros')
        print("grid_sample with zeros padding ok")
    except RuntimeError as e:
        print(f"grid_sample failed with zeros padding: {e}")
else:
    print("CUDA is not available. Skipping test.")