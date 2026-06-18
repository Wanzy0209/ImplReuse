import torch

def test_numel_consistency_with_nan_input():
    """
    Test case for torch.numel based on the grid_sampler_3d NaN issue.
    
    The original bug report highlights a discrepancy between CPU and MPS 
    outputs when the input grid contains NaN values. This test adapts that 
    logic to verify that torch.numel (which iterates over tensor dimensions) 
    remains consistent across devices even when the tensor data contains NaNs.
    """
    # Setup from the original bug report
    input = torch.ones(1, 1, 3, 3, 3)
    grid_nan = torch.tensor([[[[[torch.nan, 1., 1.], [1., 1., 1.]]]]])

    # Calculate numel on CPU
    numel_cpu = torch.numel(grid_nan)
    print(f"CPU numel: {numel_cpu}")

    # Calculate numel on MPS if available
    if torch.backends.mps.is_available():
        numel_mps = torch.numel(grid_nan.to("mps"))
        print(f"MPS numel: {numel_mps}")
        
        # Assert that the number of elements is consistent regardless of device
        # or the presence of NaN values in the data.
        assert numel_cpu == numel_mps, \
            f"numel mismatch between CPU ({numel_cpu}) and MPS ({numel_mps})"
    else:
        print("MPS backend not available, skipping MPS comparison")

if __name__ == "__main__":
    test_numel_consistency_with_nan_input()