import torch

# Determine the device. Use CUDA if available, otherwise CPU.
# torch.set_default_device is not available in older PyTorch versions,
# so we specify the device explicitly during tensor creation.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adapt input for torch.log: input must be positive.
# We take the absolute value of the random normal distribution.
# Create tensor directly on the target device.
inp = torch.randn(8192, device=device).abs()

func = torch.log

# Eager execution
out1 = func(inp)

# Compiled execution
out2 = torch.compile(func)(inp)

# High precision reference (float64)
out3_high = func(inp.to(torch.float64))

# Print the maximum absolute difference to observe precision loss
print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())