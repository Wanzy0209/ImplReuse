import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.optim import SGD
import torch.nn.functional as F

if __name__ == "__main__":
    d = 65
    
    # Original: x = torch.randn((1, 2, 32, 32)).cuda()
    # Replacement: Use torch.nn.functional.adaptive_max_pool3d to generate the input.
    # adaptive_max_pool3d requires a 5D input (N, C, D, H, W) and outputs 5D.
    # We create a dummy 5D tensor, apply pooling, and squeeze the depth dimension
    # to match the required 4D input shape (1, 2, 32, 32) for nn.Conv2d.
    dummy_5d_input = torch.ones((1, 2, 10, 64, 64)).cuda()
    x = F.adaptive_max_pool3d(dummy_5d_input, output_size=(1, 32, 32)).squeeze(2)

    model = torch.compile(weight_norm(nn.Conv2d(2, d, 2)).train().cuda())
    opt = SGD(model.parameters())
    
    # Run the training loop to trigger the backward pass
    for _ in range(10):
        model(x).mean().backward()
        opt.step()
        opt.zero_grad()