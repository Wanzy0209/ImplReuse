```python
import tensorflow as tf
import numpy as np

# PyTorch input shape is (N, C, D, H, W) = (1, 1, 3, 3, 3)
# TensorFlow GridSampler3D expects (N, D, H, W, C)
input_tensor = tf.ones((1, 3, 3, 3, 1), dtype=tf.float32)

# PyTorch grid is (N, D, H, W, 3).
# The source code had 6 dimensions (extra bracket), corrected here to 5D to match the output logic.
# Shape (1, 1, 1, 2, 3)
grid_nan = tf.constant([[[[np.nan, 1., 1.], [1., 1., 1.]]]], dtype=tf.float32)

# PyTorch: interpolation_mode=0 (bilinear), padding_mode=0 (zeros), align_corners=True
out_cpu = tf.raw_ops.GridSampler3D(
    input=input_tensor,
    grid=grid_nan,
    interpolation_mode='bilinear',
    padding_mode='zeros',
    align_corners=True
)

# Simulate MPS (GPU) execution
# Note: TensorFlow uses '/GPU:0' for GPU devices
with tf.device('/GPU:0'):
    out_gpu = tf.raw_ops.GridSampler3D(
        input=input_tensor,
        grid=grid_nan,
        interpolation_mode='