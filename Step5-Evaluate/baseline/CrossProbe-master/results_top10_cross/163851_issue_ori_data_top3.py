```python
import tensorflow as tf

# PyTorch input shape (N, C, D, H, W) -> (1, 1, 3, 3, 3)
# TensorFlow GridSampler3D expects (N, D, H, W, C) -> (1, 3, 3, 3, 1)
input = tf.ones((1, 1, 3, 3, 3))
input = tf.transpose(input, [0, 2, 3, 4, 1])

# Grid shape (N, D, H, W, 3) is consistent between PyTorch and TensorFlow
grid_nan = tf.constant([[[[[tf.float32.nan, 1., 1.],[1., 1., 1.]]]]], dtype=tf.float32)

# PyTorch args: interpolation_mode=0 (bilinear), padding_mode=0 (zeros), align_corners=True
# TensorFlow uses string arguments for modes
out_cpu = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

# PyTorch "mps" device maps to TensorFlow "/GPU:0"
# We use tf.device context to place operations and tensors on the GPU
with tf.device('/GPU:0'):
    # Ensure inputs are on the correct device
    input_gpu = tf.identity(input)
    grid_gpu = tf.identity(grid_nan)
    out_mps = tf.raw_ops.GridSampler3D(input=input_gpu, grid=grid_gpu, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

print("CPU:", tf.reshape(out_cpu, [-1])) # tensor([nan, 1.])
print("MPS:", tf.reshape(out_mps, [-1])) # tensor([1., 1.], device='mps:0')
```