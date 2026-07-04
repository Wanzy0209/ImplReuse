import torch

# Replicate the setup from the original bug report
# torch.set_default_device was introduced in PyTorch 2.1.
# For compatibility, we explicitly specify the device during tensor creation.
device = torch.device('cuda')

# torch.add requires two inputs, unlike torch.exp
inp1 = torch.randn(8192, device=device)
inp2 = torch.randn(8192, device=device)

# Adapt the function to the similar API
func = torch.add

# Eager execution
out1 = func(inp1, inp2)

# Compiled execution
out2 = torch.compile(func)(inp1, inp2)

# High precision reference (float64)
out3_high = func(inp1.to(torch.float64), inp2.to(torch.float64))

# Check precision differences
print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())