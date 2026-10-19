```python
import tensorflow as tf

# Conversion: PyTorch uses explicit device objects. TF uses device strings or context managers.
# We assume a GPU is available as per the original code's intent.
device = '/GPU:0'

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    
    # Case 1: bfloat16 + channels_last  => NaN
    # Conversion: TF uses NHWC (channels_last) format by default.
    # We create the tensor directly in NHWC format (N, H, W, C) to match the PyTorch channels_last layout.
    with tf.device(device):
        x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

        # Case 2: float32 + channels_last => illegal memory access
        # x = tf.random.normal((N, H, W, C), dtype=tf.float32)  # default float32

        # Convert to NHWC channels_last layout
        # Conversion: TF tensors are immutable and typically created in NHWC.
        # No explicit conversion needed here as we created it as NHWC above.
        # x = x # No-op

        # Uncommenting the line below avoids NaNs
        # x = x.contiguous()
        # Conversion: TF tensors are immutable; contiguity is handled by the framework.

        # Conversion: TF does not expose is_contiguous() or stride() in the high-level API.
        # We print the shape instead to verify dimensions.
        print(f"Input tensor shape: {x.shape}")
        # print(f"Input stride: {x.stride()}")

        # Conversion: PyTorch padding=1 for kernel_size=3 is equivalent to 'same' padding in TF.
        pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        y = pool(x)

        # Conversion: Use tf.math.is_nan and tf.reduce_any
        has_nan = tf.reduce_any(tf.math.is_nan(y))
        print(f"Output contains NaN? {has_nan}")

        # Conversion: Use tf.math.is_inf and tf.reduce_any
        has_inf = tf.reduce_any(tf.math.is_inf(y))
        print(f"Output contains Inf? {has_inf}")

        # Conversion: Use tf.reduce_min and tf.reduce_max
        print(f"Stats: min={tf.reduce_min(y)}, max={tf.reduce_max(y)}")

        if has_nan:
            print("Detected NaNs in MaxPool output!")

if __name__ == "__main__":
    test_maxpool_nan()
```