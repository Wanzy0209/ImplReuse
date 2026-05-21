import torch

# Ensure CUDA is available if the test is meant to run on GPU
if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    # Fallback to CPU if CUDA is not available, though the original issue was CUDA-specific
    pass

# Define the shape for the random tensor generation
shape = (8192,)

# Set the function to the similar API: torch.rand
func = torch.rand

# Set a manual seed to ensure reproducibility for random operations
torch.manual_seed(0)

# Run eager execution (float32)
out1 = func(shape)

# Reset seed to ensure the compiled version generates the same "random" numbers
torch.manual_seed(0)

# Run compiled execution (float32)
out2 = torch.compile(func)(shape)

# Reset seed for the high precision reference
torch.manual_seed(0)

# Run eager execution with float64 for reference
out3_high = func(shape, dtype=torch.float64)

# Print the maximum absolute difference to check for consistency/precision
# For random operations, out1 and out2 should be identical if compilation preserves RNG logic.
# out3_high vs out1 checks the precision difference of the RNG implementation across dtypes.
print("Max diff (eager vs compiled):", (out1 - out2).abs().max())
print("Max diff (high vs eager):", (out3_high - out1).abs().max())
print("Max diff (high vs compiled):", (out3_high - out2).abs().max())