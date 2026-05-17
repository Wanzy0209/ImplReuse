import torch
import torch.nn.functional as F

def test_poisson_nll_loss_mps():
    """
    Test case for torch.nn.functional.poisson_nll_loss on the MPS device.
    This test is derived from a bug report where grid_sampler_3d was not 
    implemented for MPS. We verify if poisson_nll_loss works on MPS.
    """
    # Check if MPS is available (Mac Metal)
    if not torch.backends.mps.is_available():
        print("MPS device not found. Skipping test.")
        return

    device = torch.device("mps")

    # Generate random input and target tensors on MPS
    # input: expectation of underlying Poisson distribution
    input_tensor = torch.rand(10, 5, device=device)
    # target: random sample ~ Poisson(input)
    target_tensor = torch.poisson(input_tensor).to(device=device)

    try:
        # Call the similar API: torch.nn.functional.poisson_nll_loss
        # This replaces the call to grid_sample from the original bug report
        loss = F.poisson_nll_loss(input_tensor, target_tensor)

        # Verify the output is valid and on the correct device
        assert loss is not None, "Loss output should not be None"
        assert loss.device.type == 'mps', "Loss output should be on MPS device"
        assert torch.isfinite(loss), "Loss should be finite"

        print(f"Test Passed. poisson_nll_loss computed successfully on MPS. Loss: {loss.item()}")

    except NotImplementedError as e:
        # This mimics the error seen in the original bug report for grid_sample
        print(f"NotImplementedError: The operator 'aten::poisson_nll_loss' is not currently implemented for the MPS device.")
        print(f"Error details: {e}")
        raise

if __name__ == "__main__":
    test_poisson_nll_loss_mps()