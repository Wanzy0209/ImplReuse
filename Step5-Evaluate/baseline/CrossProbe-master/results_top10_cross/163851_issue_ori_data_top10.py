```python
import tensorflow as tf

# Conversion: PyTorch input shape (N, C, D, H, W) -> TF input shape (N, D, H, W, C)
input = tf.ones((1, 3, 3, 3, 1))
grid_nan = tf.constant([[[[[float('nan'), 1., 1.],[1., 1., 1.]]]]])

# Conversion: torch.grid_sampler_3d -> tf.raw_ops.GridSampler3D
# Args: mode 0='bilinear', padding_mode 0='zeros', align_corners=True
with tf.device("/CPU:0"):
    out_cpu = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

# Conversion: 'mps' device -> 'gpu' device
with tf.device("/GPU:0"):
    out_mps = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

print("CPU:", tf.reshape(out_cpu, [-1])) # tensor([nan, 1.])
print("MPS:", tf.reshape(out_mps, [-1])) # tensor([1., 1.], device='mps:0')
```