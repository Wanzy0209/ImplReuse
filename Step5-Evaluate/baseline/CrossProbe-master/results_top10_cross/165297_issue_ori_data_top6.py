```python
import tensorflow as tf

# PyTorch: device = torch.device("cuda")
# TensorFlow: Use GPU device string
device = '/GPU:0'

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    # Case 1: bfloat16 + channels_last  => NaN
    # Conversion: PyTorch uses NCHW, TF uses NHWC. We create the tensor in NHWC directly.
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: torch.bfloat16 -> tf.bfloat16
    with tf.device(device):
        x = tf.random.normal([N, H, W, C], dtype=tf.bfloat16)

    # Case 2: float32 + channels_last => illegal memory access
    # x = torch.randn(N, C, H, W, device=device)  # default float32
    # x = tf.random.normal([N, H, W, C], dtype=tf.float32)

    # Convert to NHWC channels_last layout
    # PyTorch: x = x.to(memory_format=torch.channels_last)
    # TensorFlow: Data is already in NHWC format (channels_last) by default.
    # x = x # No-op

    # Uncommenting the line below avoids NaNs
    # x = x.contiguous()
    # TensorFlow: Tensors are immutable; contiguity is managed automatically.

    # PyTorch: print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
    # TensorFlow: TF does not expose memory contiguity checks in the same way.
    print(f"Input tensor shape: {x.shape}")
    # PyTorch: print(f"Input stride: {x.stride()}")
    # TensorFlow: Strides are internal implementation details.
    # print(f"Input stride: {x.strides}")

    # PyTorch: pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
    # Conversion: nn.MaxPool2d -> tf.keras.layers.MaxPooling2D
    # Conversion: padding=1 (with kernel 3, stride 2) -> padding='same'
    # Conversion: channels_last -> data_format='channels_last'
    pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last')

    with tf.device(device):
        y = pool(x)

    # PyTorch: print(f"Output contains NaN? {torch.isnan(y).any().item()}")
    # Conversion: torch.isnan -> tf.math.is_nan, .any() -> tf.reduce_any, .item() -> .numpy()
    has_nan = tf.reduce_any(tf.math.is_nan(y))
    print(f"Output contains NaN? {has_nan}")

    # PyTorch: print(f"Output contains Inf? {torch.isinf(y).any().item()}")
    # Conversion: torch.isinf -> tf.math.is_inf
    has_inf = tf.reduce_any(tf.math.is_inf(y))
    print(f"Output contains Inf? {has_inf}")

    # PyTorch: print(f"Stats: min={y.min().item()}, max={y.max().item()}")
    # Conversion: .min() -> tf.reduce_min, .max() -> tf.reduce_max
    min_val = tf.reduce_min(y)
    max_val = tf.reduce_max(y)
    print(f"Stats: min={min_val.numpy()}, max={max_val.numpy()}")

    # PyTorch: if torch.isnan(y).any():
    if has_nan:
        print("Detected NaNs in MaxPool output!")

if __name__ == "__main__":
    test_maxpool_nan()
```