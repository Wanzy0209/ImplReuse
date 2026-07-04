```python
import tensorflow as tf

# Conversion: PyTorch input format is (N, C, D, H, W), TensorFlow is (N, D, H, W, C)
input = tf.ones((1, 3, 3, 3, 1))

# Conversion: torch.tensor -> tf.constant
# Grid shape (1, 1, 1, 2, 3) corresponds to (N, D_out, H_out, W_out, 3)
grid_nan = tf.constant([[[[[float('nan'), 1., 1.],[1., 1., 1.]]]]])

# Conversion: torch.grid_sampler_3d -> tf.raw_ops.GridSampler3D
# Args: interpolation_mode=0 (bilinear), padding_mode=0 (zeros), align_corners=True
out_cpu = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

# Conversion: PyTorch MPS (Metal) is not supported by TensorFlow.
# We use GPU as the closest equivalent for device-specific execution.
# Note: This requires a GPU-enabled TensorFlow build.
with tf.device("/GPU:0"):
    out_mps = tf.raw_ops.GridSampler3D(input=input, grid=grid_nan, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)

# Conversion: flatten() -> tf.reshape(..., [-1])
print("CPU:", tf.reshape(out_cpu, [-1])) # tensor([nan, 1.])
print("MPS:", tf.reshape(out_mps, [-1])) # tensor([1., 1.], device='mps:0')
```