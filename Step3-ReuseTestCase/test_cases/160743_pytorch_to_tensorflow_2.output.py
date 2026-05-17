import tensorflow as tf
import numpy as np

# Replicate the seed setup from the original PyTorch test case
tf.random.set_seed(0)
np.random.seed(0)

# Input data mimicking the shape (4, 6, 7) from the PyTorch bug report
data = tf.random.normal((4, 6, 7))

# Define segment parameters. 
# Since AvgPool2d performs a spatial reduction, we simulate a reduction 
# by assigning random segment IDs to the tensor elements.
num_segments = 5
segment_ids = tf.random.uniform((4, 6, 7), minval=0, maxval=num_segments, dtype=tf.int32)

# Run the TensorFlow operation (similar to running model(x.to("mps")))
# This operation computes the sum along segments divided by sqrt(N)
out_tf = tf.math.unsorted_segment_sqrt_n(data, segment_ids, num_segments)

# Calculate the expected output manually to verify correctness (similar to model(x) on CPU)
# This acts as the ground truth logic: Sum / sqrt(Count)
out_expected = np.zeros(num_segments)
counts = np.zeros(num_segments)

flat_data = data.numpy().flatten()
flat_ids = segment_ids.numpy().flatten()

for val, sid in zip(flat_data, flat_ids):
    out_expected[sid] += val
    counts[sid] += 1

# Apply the sqrt(N) divisor logic
# Note: If a segment is empty (count is 0), the output should be 0, matching TF behavior
mask = counts > 0
out_expected[mask] /= np.sqrt(counts[mask])

# Verify the output
if not np.allclose(out_tf.numpy(), out_expected, atol=1e-5):
    print("Output does not match!")
    print("TensorFlow Output:\n", out_tf.numpy())
    print("Expected Output:\n", out_expected)
else:
    print("Test Passed.")