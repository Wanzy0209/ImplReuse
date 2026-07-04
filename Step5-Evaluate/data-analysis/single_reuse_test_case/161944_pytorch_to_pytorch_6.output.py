import torch

# Fix: torch.set_default_device is not available in older PyTorch versions.
# We explicitly set the device for the input tensor instead.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

inp = torch.randn(8192, device=device)

# Adapted to use torch.mul (element-wise multiplication)
func = lambda x: torch.mul(x, x)

out1 = func(inp)
out2 = torch.compile(func)(inp)
out3_high = func(inp.to(torch.float64))

print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())