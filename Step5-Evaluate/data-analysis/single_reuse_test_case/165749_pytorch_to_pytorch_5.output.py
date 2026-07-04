import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.optim import SGD

if __name__ == "__main__":
    d = 65
    x = torch.randn((1, 2, 32, 32)).cuda()
    
    # Adapted model to include torch.any, keeping the context of weight_norm and Conv2d
    class AnyConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = weight_norm(nn.Conv2d(2, d, 2))
        
        def forward(self, x):
            x = self.conv(x)
            # Use torch.any here. We cast to float to allow a backward pass.
            # This tests if torch.compile handles torch.any correctly in the graph.
            return x.mean() + torch.any(x > 0).float()

    model = torch.compile(AnyConvModel().train().cuda())
    opt = SGD(model.parameters())
    
    # Run training loop to verify backward pass with torch.any
    try:
        for _ in range(10):
            loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print("Test passed: torch.any works with torch.compile and weight_norm.")
    except Exception as e:
        print(f"Test failed: {e}")
        raise