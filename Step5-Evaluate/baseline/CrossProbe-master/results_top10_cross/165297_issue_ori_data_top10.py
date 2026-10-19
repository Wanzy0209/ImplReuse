```python
import tensorflow as tf

# PyTorch: device = torch.device("cuda")
# TensorFlow: Check for GPU availability and set device string
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    # Case 1: bfloat16 + channels_last  => NaN
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: torch.bfloat16 -> tf.bfloat16
    # Conversion: Layout NCHW (PyTorch default) -> NHWC (TensorFlow default)
    # Note: The original PyTorch code converts to channels_last (NHWC) later.
    # In TensorFlow, we create the tensor directly in NHWC format.
    with tf.device(device):
        x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

        # PyTorch: x = x.to(memory_format=torch.channels_last)
        # TensorFlow: Tensors are NHWC by default, no explicit conversion needed.

        # Uncommenting the line below avoids NaNs
        # PyTorch: x = x.contiguous()
        # TensorFlow: Memory contiguity is handled automatically by the framework.

        # PyTorch: print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
        # TensorFlow: Contiguity checks are not exposed in the same way.
        print(f"Input tensor shape: {x.shape}")

        # PyTorch: print(f"Input stride: {x.stride()}")
        # TensorFlow: Strides are internal implementation details.
        
        # PyTorch: pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        # Conversion: nn.MaxPool2d -> tf.keras.layers.MaxPooling2D
        # Note: padding=1 in PyTorch with kernel=3, stride=2 corresponds to 'same' padding in TF
        pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        y = pool(x)

        # PyTorch: print(f"Output contains NaN? {torch.isnan(y).any().item()}")
        # Conversion: torch.isnan -> tf.math.is_nan
        has_nan = tf.reduce_any(tf.math.is_nan(y))
        print(f"Output contains NaN? {has_nan}")

        # PyTorch: print(f"Output contains Inf? {torch.isinf(y).any().item()}")
        # Conversion: torch.isinf -> tf.math.is_inf
        has_inf = tf.reduce_any(tf.math.is_inf(y))
        print(f"Output contains Inf? {has_inf}")

        # PyTorch: print(f"Stats: min={y.min().item()}, max={y.max().item()}")
        # Conversion: min/max -> reduce_min/reduce_max
        print(f"Stats: min={tf.reduce_min(y)}, max={tf.reduce_max(y)}")

        # PyTorch: if torch.isnan(y).any():
        if has_nan:
            print("Detected NaNs in MaxPool output!")

test_maxpool_nan()
```