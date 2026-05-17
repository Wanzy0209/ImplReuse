import torch

# Reproduce the large tensor scenario (>4GB)
# Shape: 2 x (2^31 + 5) elements
# Size: 2 * (2^31 + 5) bytes = 4GB + 10 bytes
dim1 = 2
dim2 = (1 << 31) + 5

# Create the tensor on MPS
# Note: torch.ones uses torch.full internally, which is subject to the reported bug
# (values might not be 1). However, we will overwrite specific values to test argmax.
a = torch.ones(dim1, dim2, dtype=torch.int8, device='mps')

# To test argmax effectively on a large tensor, we need a known maximum at a known location.
# We set the first element to 0 and the last element to 2.
# This ensures the maximum is unique and located at the very end of the buffer.
a[0, 0] = 0
a[1, -1] = 2

# Calculate the expected linear index of the maximum value [1, -1]
# Linear index = (row_index * num_cols) + col_index
# row_index = 1
# col_index = dim2 - 1
# Expected index = 1 * dim2 + (dim2 - 1) = 2 * dim2 - 1
expected_index = 2 * dim2 - 1

# Perform argmax
result_index = torch.argmax(a)

# Verify the result
# This checks if argmax can handle indices exceeding 32-bit integer range (2^32 - 1)
print(f"Expected index: {expected_index}")
print(f"Result index:   {result_index}")

assert result_index == expected_index, (
    f"torch.argmax failed on large MPS tensor. "
    f"Expected index {expected_index}, but got {result_index}."
)