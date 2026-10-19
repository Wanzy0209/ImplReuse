import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.optim import SGD
import torch.nn.functional as F

if __name__ == "__main__":
    d = 65
    
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # We create a base tensor to apply the pooling operation to generate the input x.
    # Using torch.ones to create a base tensor of a larger size, then pooling down to (32, 32).
    base_input = torch.ones(1, 2, 64, 64).cuda()
    x = F.adaptive_max_pool2d(base_input, (32, 32))
    
    # Verify the shape matches the original requirement
    assert x.shape == (1, 2, 32, 32), f"Input shape mismatch: {x.shape}"

    model = torch.compile(weight_norm(nn.Conv2d(2, d, 2)).train().cuda())
    opt = SGD(model.parameters())

    try:
        # The bug triggers during the backward pass
        model(x).mean().backward()
        opt.step()
        opt.zero_grad()
        print("Test passed: Backward pass completed without error.")
    except RuntimeError as e:
        if "saved_v must be contiguous" in str(e):
            print(f"Test failed: Bug reproduced - {e}")
            raise
        else:
            raise