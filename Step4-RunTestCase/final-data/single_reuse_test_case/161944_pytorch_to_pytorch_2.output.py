import torch

# torch.set_default_device was introduced in PyTorch 2.1.
# For compatibility with older versions, we explicitly specify the device
# during tensor creation instead of setting a global default.
device = 'cuda'

inp = torch.randn(8192, device=device)

# Replace torch.exp with the similar API torch.tanh
func = torch.tanh

# Eager execution (float32)
out1 = func(inp)

# Compiled execution (float32)
out2 = torch.compile(func)(inp)

# High precision reference (float64)
out3_high = func(inp.to(torch.float64))

# Check precision differences
print("Max diff (Eager vs High):", (out3_high - out1).abs().max())
print("Max diff (Compiled vs High):", (out3_high - out2).abs().max())