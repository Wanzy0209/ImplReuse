import torch

# Replicate the setup from the original bug report
# Fix: torch.set_default_device is not available in older PyTorch versions.
# Instead, we determine the device and pass it to tensor creation.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("CUDA not available, running on CPU")

# Generate inputs for division (requires two tensors)
# Pass the device explicitly to ensure tensors are on the correct device
inp1 = torch.randn(8192, device=device)
inp2 = torch.randn(8192, device=device)

# Adapt the function to torch.div
func = torch.div

# 1. Eager execution (float32)
out1 = func(inp1, inp2)

# 2. Compiled execution (float32)
out2 = torch.compile(func)(inp1, inp2)

# 3. High precision reference (float64)
out3_high = func(inp1.to(torch.float64), inp2.to(torch.float64))

# Compare the maximum absolute difference against the high precision reference
# This checks if the compiled version introduces significant numerical deviations
print((out3_high - out1).abs().max())
print((out3_high - out2).abs().max())