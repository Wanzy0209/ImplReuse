import torch
from torch import nn
from torch.optim import SGD

if __name__ == "__main__":
    # Setup similar to the original bug report
    x = torch.randn((1, 2, 32, 32)).cuda()
    
    # Define a model that uses torch.all
    # Since torch.all is non-differentiable, we include a differentiable path
    # to maintain the training loop structure of the original test case.
    class AllModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 2)
        
        def forward(self, x):
            x = self.conv(x)
            # Use torch.all to verify a condition on the tensor
            # This tests the compilation of the torch.all polyfill
            check = torch.all(x > -1.0)
            return x, check

    model = torch.compile(AllModel().train().cuda())
    opt = SGD(model.parameters())
    
    for _ in range(10):
        out, all_check = model(x)
        # Perform backward pass on the differentiable output
        out.mean().backward()
        opt.step()
        opt.zero_grad()
        
        # Assert that torch.all returns the expected boolean tensor
        assert isinstance(all_check, torch.Tensor)
        assert all_check.dtype == torch.bool