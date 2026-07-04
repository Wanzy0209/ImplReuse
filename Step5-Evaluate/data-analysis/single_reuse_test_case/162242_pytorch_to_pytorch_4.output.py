import numpy as np
import torch


def test_floor_deterministic():
    # Setup input data with non-integer values to make floor meaningful
    # Using the same shape as the original test for consistency
    input_data = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4]).cuda() + 0.5
    
    # Expected result: floor of (0.5, 1.5, ...) is (0, 1, ...)
    gt_res = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4]).cpu().numpy()
    
    # Expected gradient: derivative of floor is 0
    gt_grad = np.zeros((2, 3, 4), dtype=np.float32)

    for i in range(1000):
        torch.cuda.empty_cache()
        
        # Create tensor for this iteration
        x = input_data.clone().detach()
        x.requires_grad = True

        # Call the API: torch.floor
        res = torch.floor(x)
        
        # Compute gradients
        res.backward(torch.ones_like(res))

        print(f"Test {i + 1}/{1000}")
        
        # Assert determinism by checking against ground truth
        np.testing.assert_allclose(res.cpu().detach().numpy(), gt_res)
        np.testing.assert_allclose(x.grad.cpu().numpy(), gt_grad)


if __name__ == "__main__":
    test_floor_deterministic()