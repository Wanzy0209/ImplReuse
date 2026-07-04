import torch

# Check for CUDA availability to match the original test environment
if not torch.cuda.is_available():
    print("CUDA is not available. This test requires a GPU.")
    exit()

# Create a tensor with channels_last memory format, similar to the bug report
x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)

# Apply the similar API: torch.atleast_1d
# Note: atleast_1d on a 4-D tensor returns the tensor itself, but we verify memory format preservation.
y = torch.atleast_1d(x)

# Verify the results
# The original bug report checked equality and storage to detect memory ordering changes.
is_equal = torch.equal(x, y)
is_channels_last = y.is_contiguous(memory_format=torch.channels_last)

print('Equal: {}\n x storage:\n{}\n y storage:\n{}\n'.format(is_equal, x.storage(), y.storage()))

# Assertions to ensure the API behaves as expected regarding memory format
assert is_equal, "Input and output tensors should be equal"
assert is_channels_last, "Output tensor should preserve channels_last memory format"