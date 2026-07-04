```python
import tensorflow as tf

# x_cpu = torch.randn((5, ), requires_grad=True)
# Conversion: tf.random.normal creates a tensor with values from a normal distribution.
# requires_grad=True is implicit in TF2 eager execution.
x_cpu = tf.random.normal((5, ), dtype=tf.float32)

# src_cpu = torch.randn((5,), requires_grad=True)
src_cpu = tf.random.normal((5,), dtype=tf.float32)

# index = torch.randint(0, 5, (5,))
# Conversion: tf.random.uniform with integer dtype mimics randint.
index = tf.random.uniform((5,), minval=0, maxval=5, dtype=tf.int32)

# x_mps = x_cpu.detach().to("mps").requires_grad_(True)
# Conversion: Move to GPU. tf.stop_gradient mimics detach().
with tf.device('/GPU:0'):
    x_mps = tf.stop_gradient(tf.identity(x_cpu))
    # src_mps = src_cpu.detach().to("mps").requires_grad_(True)
    src_mps = tf.stop_gradient(tf.identity(src_cpu))
    # index_mps = index.to('mps')
    index_mps = tf.identity(index)

# out_cpu = torch.index_copy(x_cpu, 0, index, src_cpu)
# Conversion: tf.tensor_scatter_nd_update is the equivalent operation.
# Indices must be expanded to shape (N, rank) for TF, here (5, 1).
indices_expanded_cpu = tf.expand_dims(index, axis=-1)
out_cpu = tf.tensor_scatter_nd_update(x_cpu, indices_expanded_cpu, src_cpu)

# out_mps = torch.index_copy(x_mps, 0, index_mps, src_mps)
with tf.device('/GPU:0'):
    indices_expanded_mps = tf.expand_dims(index_mps, axis=-1)
    out_mps = tf.tensor_scatter_nd_update(x_mps, indices_expanded_mps, src_mps)

# torch.testing.assert_close(out_cpu, out_mps.cpu())
# Conversion: tf.debugging.assert_near checks for numerical closeness.
tf.debugging.assert_near(out_cpu, out_mps)
```