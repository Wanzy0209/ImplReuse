```python
import tensorflow as tf

# TensorFlow uses NHWC (channels_last) memory format by default.
# We define the device string to mimic torch.device("cuda").
device = '/GPU:0'

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    
    # Case 1: bfloat16 + channels_last  => NaN
    # Note: In TensorFlow, the default data format is NHWC (channels_last).
    # We create the tensor with shape (N, H, W, C) directly.
    with tf.device(device):
        x = tf.random.normal([N, H, W, C], dtype=tf.bfloat16)

        # Case 2: float32 + channels_last => illegal memory access
        # x = tf.random.normal([N, H, W, C], dtype=tf.float32)  # default float32

        # Convert to NHWC channels_last layout
        # PyTorch: x = x.to(memory_format=torch.channels_last)
        # TensorFlow: Tensors are created in NHWC by default, so no explicit conversion needed.
        
        # Uncommenting the line below avoids NaNs
        # x = tf.identity(x) # TF equivalent of ensuring a fresh copy/contiguousness if needed

        print(f"Input tensor: shape={x.shape}")
        # TF does not expose strides/contiguous checks in the same way for user debugging in eager mode.
        # print(f"Input stride: {x.stride()}") 

        # PyTorch: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # TensorFlow: padding='same' is equivalent to padding=1 for kernel_size=3 to maintain dimensions.
        pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        y = pool(x)

        print(f"Output contains NaN? {tf.reduce_any(tf.math.is_nan(y))}")
        print(f"Output contains Inf? {tf.reduce_any(tf.math.is_inf(y))}")
        print(f"Stats: min={tf.reduce_min(y)}, max={tf.reduce_max(y)}")

        if tf.reduce_any(tf.math.is_nan(y)):
            print("Detected NaNs in MaxPool output!")

if __name__ == "__main__":
    # Check for GPU availability to avoid error if run on CPU-only machine
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            test_maxpool_nan()
        except RuntimeError as e:
            print(e)
    else:
        print("GPU not found, skipping test.")
```