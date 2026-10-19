```python
import tensorflow as tf

# device = torch.device("cuda")
# TensorFlow manages device placement automatically (e.g., GPU if available).
# We proceed assuming the default device context.

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    # Case 1: bfloat16 + channels_last  => NaN
    # PyTorch default is NCHW, so we create (N, C, H, W)
    x = tf.random.normal((N, C, H, W), dtype=tf.bfloat16)

    # Case 2: float32 + channels_last => illegal memory access
    # x = tf.random.normal((N, C, H, W), dtype=tf.float32)  # default float32


    # Convert to NHWC channels_last layout
    # PyTorch: x.to(memory_format=torch.channels_last)
    # TF: Transpose dimensions from NCHW to NHWC
    x = tf.transpose(x, [0, 2, 3, 1])

    # Uncommenting the line below avoids NaNs
    # x = x.contiguous()
    # TensorFlow tensors are immutable; memory layout is managed internally.

    # PyTorch: print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    # TF does not expose is_contiguous(). We print shape and data format instead.
    print(f"Input tensor: shape={x.shape}, data_format=NHWC")
    # PyTorch: print(f"Input stride: {x.stride()}")
    # TF does not expose stride() directly in the public API.
    print(f"Input shape: {x.shape}")

    # pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
    # TF: tf.keras.layers.MaxPooling2D. padding='same' approximates padding=1 for this kernel/stride combo.
    pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

    y = pool(x)

    # print(f"Output contains NaN? {torch.isnan(y).any().item()}")
    print(f"Output contains NaN? {tf.reduce_any(tf.math.is_nan(y)).numpy()}")
    # print(f"Output contains Inf? {torch.isinf(y).any().item()}")
    print(f"Output contains Inf? {tf.reduce_any(tf.math.is_inf(y)).numpy()}")
    # print(f"Stats: min={y.min().item()}, max={y.max().item()}")
    print(f"Stats: min={tf.reduce_min(y).numpy()}, max={tf.reduce_max(y).numpy()}")

    # if torch.isnan(y).any():
    if tf.reduce_any(tf.math.is_nan(y)).numpy():
        print("Detected NaNs in MaxPool output!")

test_maxpool_nan()
```