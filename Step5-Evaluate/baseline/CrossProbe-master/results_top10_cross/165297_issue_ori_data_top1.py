```python
import tensorflow as tf

# Conversion: torch.device("cuda") -> Check for GPU availability
# TF uses the first available GPU by default if operations support it.
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    # Case 1: bfloat16 + channels_last  => NaN
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: PyTorch uses NCHW, TF uses NHWC. Since source converts to channels_last (NHWC),
    # we initialize TF tensor directly in NHWC format.
    with tf.device(device):
        x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

    # Case 2: float32 + channels_last => illegal memory access
    # x = torch.randn(N, C, H, W, device=device)  # default float32
    # Conversion: x = tf.random.normal((N, H, W, C), dtype=tf.float32)

    # Convert to NHWC channels_last layout
    # x = x.to(memory_format=torch.channels_last)
    # Conversion: TF tensors are created in NHWC format by default, no conversion needed.

    # Uncommenting the line below avoids NaNs
    # x = x.contiguous()
    # Conversion: TF tensors are immutable and contiguous by default.

    # print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    # Conversion: TF does not expose memory contiguity checks in the Python API.
    print(f"Input tensor shape: {x.shape}")

    # print(f"Input stride: {x.stride()}")
    # Conversion: TF does not expose strides directly in the same way.
    print(f"Input dtype: {x.dtype}")

    # Conversion: nn.MaxPool2d -> tf.keras.layers.MaxPooling2D
    # padding=1 in PyTorch (kernel=3, stride=2) corresponds to 'same' padding in TF
    pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
    
    with tf.device(device):
        y = pool(x)

    # Conversion: torch.isnan -> tf.math.is_nan, .any() -> tf.reduce_any
    print(f"Output contains NaN? {tf.reduce_any(tf.math.is_nan(y)).numpy()}")
    # Conversion: torch.isinf -> tf.math.is_inf
    print(f"Output contains Inf? {tf.reduce_any(tf.math.is_inf(y)).numpy()}")
    
    # Conversion: .min()/.max() -> tf.reduce_min/reduce_max
    print(f"Stats: min={tf.reduce_min(y).numpy()}, max={tf.reduce_max(y).numpy()}")

    if tf.reduce_any(tf.math.is_nan(y)):
        print("Detected NaNs in MaxPool output!")

test_maxpool_nan()
```