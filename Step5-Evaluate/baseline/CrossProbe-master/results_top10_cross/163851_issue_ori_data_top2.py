```python
import tensorflow as tf

# Conversion: PyTorch input shape (N, C, D, H, W) -> TensorFlow input shape (N, D, H, W, C)
input = tf.ones((1, 3, 3, 3, 1))

# Conversion: torch.tensor -> tf.constant
# Grid shape (N, D, H, W, 3) is consistent between PyTorch and TensorFlow for grid sampling
grid_nan = tf.constant([[[[[tf.nan, 1., 1.],[1., 1., 1.]]]]])

# Conversion: torch.grid_sampler_3d -> tf.raw_ops.GridSampler3D
# Args: interpolation_mode 0='bilinear', padding_mode 0='zeros', align_corners=True
out_cpu = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode="bilinear", padding_mode="zeros", align_corners=True)

# Conversion: .to("mps") -> tf.device('/GPU:0')
# Note: TensorFlow's behavior with NaNs is generally consistent across backends (propagating NaN),
# unlike the specific CPU/MPS discrepancy shown in the original PyTorch code.
with tf.device('/GPU:0'):
    out_mps = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode="bilinear", padding_mode="zeros", align_corners=True)

print("CPU:", tf.reshape(out_cpu, [-1])) # flatten
print("MPS:", tf.reshape(out_mps, [-1])) # flatten
```