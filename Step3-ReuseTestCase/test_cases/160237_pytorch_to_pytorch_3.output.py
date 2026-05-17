import torch
import torch.nn as nn

def test_soft_margin_loss_on_mps():
    """
    Test case for torch.nn.SoftMarginLoss on the MPS device.
    
    Context: Issue 160237 reports a NotImplementedError for 'aten::grid_sampler_3d' 
    on the MPS device. This test verifies that the similar API, torch.nn.SoftMarginLoss,
    functions correctly on the same hardware backend (Mac Metal).
    """
    # Check if MPS is available (Mac Metal)
    if not torch.backends.mps.is_available():
        print("MPS device is not available. Skipping test.")
        return

    # Define input and target tensors
    # SoftMarginLoss expects target values to be 1 or -1
    batch_size = 4
    features = 5
    input_tensor = torch.randn(batch_size, features)
    target_tensor = torch.empty(batch_size, features).random_(2) * 2 - 1  # Generates 1 or -1

    # Move tensors to MPS device
    device = torch.device("mps")
    input_mps = input_tensor.to(device)
    target_mps = target_tensor.to(device)

    # Initialize the loss function
    loss_fn = nn.SoftMarginLoss()

    # Execute the operation
    try:
        loss = loss_fn(input_mps, target_mps)
        
        # Verify the output
        assert loss is not None, "Loss computation returned None"
        assert loss.device.type == 'mps', "Loss tensor is not on the MPS device"
        assert torch.isfinite(loss), "Loss computation resulted in NaN or Inf"
        
        print(f"Test Passed: SoftMarginLoss works on MPS. Loss value: {loss.item()}")

    except NotImplementedError as e:
        print(f"Test Failed: SoftMarginLoss is not implemented for MPS - {e}")
        raise

if __name__ == "__main__":
    test_soft_margin_loss_on_mps()