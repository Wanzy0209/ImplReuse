```python
import tensorflow as tf

# TensorFlow automatically uses GPU if available, no explicit device object needed like PyTorch
# device = torch.device("cuda")

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    # Case 1: bfloat16 + channels_last  => NaN
    # Conversion: PyTorch uses NCHW, TensorFlow uses NHWC (channels_last) by default.
    # We create the tensor with shape (N, H, W, C) to match TF's channels_last expectation.
    x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

    # Case 2: float32 + channels_last => illegal memory access
    # x = tf.random.normal((N, H, W, C), dtype=tf.float32)  # default float32

    # Convert to NHWC channels_last layout
    # Conversion: In TensorFlow, tensors are typically created in NHWC format.
    # The concept of converting memory format without copying data (non-contiguous)
    # is not exposed in the same way as PyTorch.
    # x = x # Already NHWC

    # Uncommenting the line below avoids NaNs
    # Conversion: TF tensors are immutable; operations return new tensors.
    # x = tf.identity(x)

    # Conversion: TensorFlow does not expose stride/contiguity checks in the high-level API.
    # print(f"Input tensor: contiguous={...}, channels_last={...}")
    # print(f"Input stride: {...}")

    # Conversion: nn.MaxPool2d -> tf.keras.layers.MaxPooling2D
    # padding='same' is the closest equivalent to padding=1 for kernel_size=3, stride=2
    pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
    y = pool(x)

    # Conversion: torch.isnan -> tf.math.is_nan
    # Conversion: .any().item() -> tf.reduce_any(...) (returns a 0-d tensor in eager mode)
    print(f"Output contains NaN? {tf.reduce_any(tf.math.is_nan(y))}")
    print(f"Output contains Inf? {tf.reduce_any(tf.math.is_inf(y))}")
    print(f"Stats: min={tf.reduce_min(y)}, max={tf.reduce_max(y)}")

    if tf.reduce_any(tf.math.is_nan(y)):
        print("Detected NaNs in MaxPool output!")

if __name__ == "__main__":
    test_maxpool_nan()
```