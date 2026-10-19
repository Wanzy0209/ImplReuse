```python
import tensorflow as tf

# PyTorch: device = torch.device("cuda")
# TensorFlow: Uses GPU automatically if available. We define the device string to mimic the structure.
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    
    # Case 1: bfloat16 + channels_last  => NaN
    # PyTorch: x = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=device)
    # TensorFlow: Uses NHWC format (N, H, W, C) by default. We generate in NHWC to match channels_last.
    with tf.device(device_name):
        x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

        # Case 2: float32 + channels_last => illegal memory access
        # x = torch.randn(N, C, H, W, device=device)  # default float32
        # x = tf.random.normal((N, H, W, C), dtype=tf.float32)

        # Convert to NHWC channels_last layout
        # PyTorch: x = x.to(memory_format=torch.channels_last)
        # In TensorFlow, the default data format for tensors is NHWC (channels_last), 
        # so no explicit conversion is needed here.

        # Uncommenting the line below avoids NaNs
        # PyTorch: x = x.contiguous()
        # TensorFlow tensors are immutable; contiguity is handled internally by the framework.

        # PyTorch: print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
        # TensorFlow does not expose memory format checks like is_contiguous() in the same way.
        print(f"Input tensor shape: {x.shape}, dtype: {x.dtype}")

        # PyTorch: print(f"Input stride: {x.stride()}")
        # TensorFlow abstracts memory strides from the user level.
        # print(f"Input stride: {x.strides}") # Not directly applicable in the same way

        # PyTorch: pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        # TensorFlow: padding='same' approximates padding=1 for kernel_size=3, stride=2 to maintain dimensions.
        pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        y = pool(x)

        # PyTorch: print(f"Output contains NaN? {torch.isnan(y).any().item()}")
        has_nan = tf.reduce_any(tf.math.is_nan(y))
        print(f"Output contains NaN? {has_nan}")

        # PyTorch: print(f"Output contains Inf? {torch.isinf(y).any().item()}")
        has_inf = tf.reduce_any(tf.math.is_inf(y))
        print(f"Output contains Inf? {has_inf}")

        # PyTorch: print(f"Stats: min={y.min().item()}, max={y.max().item()}")
        print(f"Stats: min={tf.reduce_min(y)}, max={tf.reduce_max(y)}")

        # PyTorch: if torch.isnan(y).any():
        if has_nan:
            print("Detected NaNs in MaxPool output!")

test_maxpool_nan()
```