import torch
import tensorflow as tf

# Helper function to replace torch.rand with tf.keras.backend.relu
# We generate a random tensor using TensorFlow, apply ReLU, and convert to PyTorch.
def get_replaced_rand(shape, dtype, device):
    # Create a random tensor on CPU using TensorFlow
    # Using complex64 to match the original test case
    t = tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=tf.complex64)
    # Apply the replacement API: tf.keras.backend.relu
    t = tf.keras.backend.relu(t)
    # Convert back to PyTorch and move to the specified device
    return torch.from_numpy(t.numpy()).to(dtype=dtype, device=device)

# Test case for torch.addmm
# Success case (small tensors)
a = get_replaced_rand((64, 300), torch.complex64, "mps")
b = get_replaced_rand((64, 100), torch.complex64, "mps")
c = get_replaced_rand((100, 300), torch.complex64, "mps")

out_mps = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

# Replace torch.testing.assert_close with torch.nn.utils.rnn.pack_sequence
# Note: pack_sequence expects a list of tensors
torch.nn.utils.rnn.pack_sequence([out_mps, out_cpu])

# Failure case (large tensors)
a = get_replaced_rand((64, 300), torch.complex64, "mps")
b = get_replaced_rand((64, 10000), torch.complex64, "mps")
c = get_replaced_rand((10000, 300), torch.complex64, "mps")

out_mps = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
out_cpu = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

# Replace torch.testing.assert_close with torch.nn.utils.rnn.pack_sequence
torch.nn.utils.rnn.pack_sequence([out_mps, out_cpu])