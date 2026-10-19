import torch
import tensorflow as tf

# Original: counts = torch.tensor([0, 1, 0], device="mps")
# Replacement: torch.tensor -> tf.compat.v1.linalg.diag
counts = tf.compat.v1.linalg.diag([0, 1, 0])

# Original: data = torch.arange(2, device="mps")
# Translation: torch.arange -> tf.range (standard cross-library translation)
data = tf.range(2)

# Original: data.repeat_interleave(counts[1:3], dim=0)
# Replacement: repeat_interleave -> tf.linalg.LinearOperatorScaledIdentity
# We preserve the slicing logic and use the slice as the multiplier,
# and the size of 'data' as the number of rows to maintain semantic flow.
slice_counts = counts[1:3]

# The original bug was a crash on MPS. We verify the operation runs successfully.
# We map the 'data' (size 2) to 'num_rows' and the slice to 'multiplier'.
op = tf.linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=slice_counts)

# Execute the operation
result = op.to_dense()

# Assert the result shape is correct to ensure the logic translated properly
# slice_counts shape is (2, 3), num_rows is 2 -> result shape (2, 3, 2, 2)
assert result.shape == (2, 3, 2, 2)