import numpy as np
import torch


def test_ceil_deterministic():
    # Define input data with decimals to test ceiling behavior
    # Using a shape similar to the original test for consistency
    input_data = np.arange(24, dtype=np.float32).reshape([2, 3, 4]) + 0.5
    
    # Ground truth for the result (ceil of x + 0.5 is x + 1.0)
    gt_res = np.ceil(input_data)
    
    # Ground truth for the gradient (gradient of ceil is 0 almost everywhere)
    gt_input_grad = np.zeros_like(input_data)

    for i in range(100):
        # Adapted call site: Create inputs tensor
        # Removed .cuda() to ensure the test runs on CPU-only environments for minimal reproducibility
        inputs = torch.tensor(input_data, dtype=torch.float32)
        
        inputs.requires_grad = True

        # Adapted call site: torch.ceil
        # Replaces torch.scatter(inputs, 1, index, src)
        res = torch.ceil(inputs)
        
        res.backward(torch.ones_like(res))

        print(f"Test {i + 1}/100")
        np.testing.assert_allclose(res.cpu().detach().numpy(), gt_res)
        np.testing.assert_allclose(inputs.grad.cpu().numpy(), gt_input_grad)


if __name__ == "__main__":
    test_ceil_deterministic()