import torch
import torch.nn.functional as F

# torch.set_default_device is not available in older PyTorch versions.
# We explicitly specify the device for the tensor instead.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

inp = torch.randn(8192, device=device)
func = F.softplus
out1 = func(inp)
out2 = torch.compile(func)(inp)
out3_high = func(inp.to(torch.float64))
print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())