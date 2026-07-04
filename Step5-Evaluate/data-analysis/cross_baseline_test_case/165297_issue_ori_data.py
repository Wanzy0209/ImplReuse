```python
import tensorflow as tf

# TensorFlow handles device placement via context managers.
# We check for GPU availability to match the torch.device("cuda") intent.
device_name = tf.test.gpu_device_name()
if not device_name:
    # Fallback to CPU if GPU is not found, though the original code assumes CUDA.
    device_name = "/CPU:0"

with tf.device(device_name):
    def test_maxpool_nan():
        # Large input tensor
        N, C, H, W = 84, 64, 512, 960
        # Case 1: bfloat16 + channels_last  => NaN
        # Conversion: PyTorch uses NCHW by default, TF uses NHWC.
        # We create the tensor directly in NHWC format to match the 'channels_last' state.
        x = tf.random.normal([N, H, W, C], dtype=tf.bfloat16)

        # Case 2: float32 + channels_last => illegal memory access
        # x = tf.random.normal([N, H, W, C], dtype=tf.float32)  # default float32


        # Convert to NHWC channels_last layout
        # Conversion: TF tensors are immutable and created in the specified shape (NHWC).
        # No explicit conversion step is needed as we created it in the target format.
        # x = x # No-op

        # Uncommenting the line below avoids NaNs
        # x = tf.identity(x) # TF equivalent of ensuring a fresh contiguous copy

        # Conversion: TF tensors are immutable, so contiguity is implicit.
        # is_contiguous() checks are not applicable in the same way.
        # print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
        print(f"Input tensor shape: {x.shape}")

        # Conversion: Strides are internal in TF and not typically exposed for user inspection.
        # print(f"Input stride: {x.stride()}")

        # Conversion: nn.MaxPool2d -> tf.keras.layers.MaxPooling2D
        # padding='same' is the TF equivalent for padding to maintain dimensions with stride.
        pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        y = pool(x)

        # Conversion: torch.isnan -> tf.math.is_nan, .any() -> tf.reduce_any
        print(f"Output contains NaN? {tf.reduce_any(tf.math.is_nan(y))}")
        print(f"Output contains Inf? {tf.reduce_any(tf.math.is_inf(y))}")
        print(f"Stats: min={tf.reduce_min(y)}, max={tf.reduce_max(y)}")

        # Conversion: torch.isnan(y).any() -> tf.reduce_any(tf.math.is_nan(y))
        if tf.reduce_any(tf.math.is_nan(y)):
            print("Detected NaNs in MaxPool output!")

    test_maxpool_nan()
```