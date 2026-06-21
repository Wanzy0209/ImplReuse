import torch

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the shape for the random tensor generation
shape = (8192,)

# Set the function to the similar API: torch.rand
func = torch.rand

# Set a manual seed to ensure reproducibility for random operations
torch.manual_seed(0)

# Run eager execution (float32)
# Explicitly pass the device argument to ensure compatibility with older PyTorch versions
out1 = func(shape, device=device)

# Reset seed to ensure the compiled version generates the same "random" numbers
torch.manual_seed(0)

# Run compiled execution (float32)
# Explicitly pass the device argument
out2 = torch.compile(func)(shape, device=device)

# Reset seed for the high precision reference
torch.manual_seed(0)

# Run eager execution with float64 for reference
# Explicitly pass the device argument
out3_high = func(shape, dtype=torch.float64, device=device)

# Print the maximum absolute difference to check for consistency/precision
# For random operations, out1 and out2 should be identical if compilation preserves RNG logic.
# out3_high vs out1 checks the precision difference of the RNG implementation across dtypes.
print("Max diff (eager vs compiled):", (out1 - out2).abs().max())
print("Max diff (high vs eager):", (out3_high - out1).abs().max())
print("Max diff (high vs compiled):", (out3_high - out2).abs().max())