import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.optim import SGD

if __name__ == "__main__":
    d = 65
    
    # Replacement: torch.randn -> torch.var_mean
    # We generate the input tensor 'x' by computing the variance of a dummy tensor
    # and expanding it to the required shape. This replaces the random data generation
    # with a statistical operation as requested.
    dummy = torch.arange(1, 1 * 2 * 32 * 32 + 1).view(1, 2, 32, 32).cuda().float()
    var, _ = torch.var_mean(dummy, dim=[1, 2, 3], keepdim=True)
    x = var.expand((1, 2, 32, 32))

    model = torch.compile(weight_norm(nn.Conv2d(2, d, 2)).train().cuda())
    opt = SGD(model.parameters())
    
    try:
        for _ in range(1000):
            model(x).mean().backward()
            opt.step()
            opt.zero_grad()
        print("Test passed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise