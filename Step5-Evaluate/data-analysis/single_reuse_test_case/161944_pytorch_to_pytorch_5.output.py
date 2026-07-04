import torch

# Check if CUDA is available, otherwise fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adapted inputs for torch.sub which requires two tensors
# Explicitly set the device for the tensors
inp1 = torch.randn(8192, device=device)
inp2 = torch.randn(8192, device=device)

func = torch.sub

# Eager execution
out1 = func(inp1, inp2)

# Compiled execution
out2 = torch.compile(func)(inp1, inp2)

# High precision reference (float64)
out3_high = func(inp1.to(torch.float64), inp2.to(torch.float64))

# Check precision differences
print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())