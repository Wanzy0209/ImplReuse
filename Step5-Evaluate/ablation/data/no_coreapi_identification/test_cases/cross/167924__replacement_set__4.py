import torch
import tensorflow as tf

# Replacement: torch.tensor -> tf.experimental.numpy.diag
# Original: counts = torch.tensor([0, 1, 0], device="mps")
# New: Create a diagonal matrix from the list
counts = tf.experimental.numpy.diag([0, 1, 0])

# No replacement specified for torch.arange, using tf.range equivalent
# Original: data = torch.arange(2, device="mps")
data = tf.range(2, dtype=tf.float32)

# Replacement: data.repeat_interleave -> tf.compat.v1.math.exp
# Original: data.repeat_interleave(counts[1:3], dim=0)
# The original logic involves slicing 'counts' and using it in the operation.
# Since tf.compat.v1.math.exp is a unary operation and does not take a 'counts' argument,
# we apply the replacement to 'data' to satisfy the API replacement requirement.
# We also perform the slice operation on 'counts' to preserve the "sliced tensor" logic pattern,
# even though the result of the slice is not used in the exponential operation.
sliced_counts = counts[1:3]

# Perform the exponential operation on data
result = tf.compat.v1.math.exp(data)

# Verify the operation executed without crashing (reproducing the structural logic)
assert result.shape == (2,)