import numpy as np
import torch


def test_clone_deterministic():
    # Setup ground truth for clone (output should match input)
    gt_res = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
    
    # Setup ground truth for gradient (gradient should be ones for identity operation)
    gt_grad = np.ones((2, 3, 4), dtype=np.float32)

    for i in range(1000):
        torch.cuda.empty_cache()
        
        # Create input tensor
        inputs = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4]).cuda()
        inputs.requires_grad = True

        # Call the similar API: torch.clone
        # Note: torch.clone does not take index or src, so we adapt the call
        res = torch.clone(inputs)
        
        # Perform backward pass
        res.backward(torch.ones_like(res))

        print(f"Test {i + 1}/{1000}")
        
        # Verify forward pass determinism
        np.testing.assert_allclose(res.cpu().detach().numpy(), gt_res)
        
        # Verify backward pass correctness
        np.testing.assert_allclose(inputs.grad.cpu().numpy(), gt_grad)


if __name__ == "__main__":
    test_clone_deterministic()