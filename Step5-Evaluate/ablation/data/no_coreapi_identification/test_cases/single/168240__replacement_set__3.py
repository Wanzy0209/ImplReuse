import torch
import torchvision
import tensorflow as tf

# Setup the model
model = torchvision.models.mobilenet_v2(weights=None)

# Replacement: torch.rand -> torch.tanh
# To make torch.tanh runnable as an input generator, we apply it to a random tensor.
x = torch.tanh(torch.randn(1, 3, 224, 224))

# Export the model
ep = torch.export.export(model, (x,))

# Get outputs
out1 = model(x)
out2 = ep.module()(x)

# Replacement: torch.testing.assert_close -> tf.compat.v1.sparse_segment_sqrt_n
# Logic: Verify that the outputs are close.
# We compute the difference and use sparse_segment_sqrt_n to check the magnitude of the difference.
diff = out1 - out2

# Convert PyTorch tensor to TensorFlow tensor
tf_diff = tf.constant(diff.numpy())

# Setup indices and segment_ids to compute the sum of differences over the whole tensor
num_elements = tf.size(tf_diff)
indices = tf.range(num_elements)
segment_ids = tf.zeros(num_elements, dtype=tf.int32)

# sparse_segment_sqrt_n computes sum(data) / sqrt(N) for the segment.
# If the models are correct, diff is ~0, so the result should be ~0.
result = tf.compat.v1.sparse_segment_sqrt_n(tf_diff, indices, segment_ids)

# Assert that the result is close to zero to verify correctness
assert result.numpy() < 1e-5