```python
import tensorflow as tf

# TensorFlow handles device placement automatically or via context managers.
# We assume GPU is available if the original code used "cuda".
# device = "/GPU:0" # Conceptual equivalent

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    
    # Case 1: bfloat16 + channels_last => NaN
    # PyTorch creates NCHW then converts to channels_last (NHWC memory layout).
    # TensorFlow uses NHWC (channels_last) as the default data format.
    # So we create the tensor directly in NHWC shape: (N, H, W, C)
    x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

    # Case 2: float32 + channels_last => illegal memory access
    # x = tf.random.normal((N, H, W, C), dtype=tf.float32) # default float32

    # In PyTorch, .to(memory_format=torch.channels_last) changes the memory layout 
    # but keeps the shape (N, C, H, W). 
    # In TensorFlow, the data format is defined by the shape. 
    # Since we created x as (N, H, W, C), it is already in "channels_last" format.
    
    # Uncommenting the line below avoids NaNs (PyTorch specific)
    # x = tf.identity(x) # TF tensors are immutable, contiguity is handled differently

    # PyTorch: contiguous checks. TF tensors are immutable, so contiguity 
    # is not a user-managed property in the same way.
    # print(f"Input tensor: contiguous=True, channels_last=True") 
    
    # PyTorch: stride check. Not directly exposed in TF Python API.
    # print(f"Input stride: Not applicable in TF")

    print(f"Input tensor shape: {x.shape}")

    # PyTorch: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # TensorFlow: tf.nn.max_pool2d
    # ksize and strides are 1D lists of length 4: [batch, height, width, channels]
    # padding='SAME' is equivalent to PyTorch padding=1 for kernel_size=3, stride=2
    y = tf.nn.max_pool2d(
        input=x, 
        ksize=3, 
        strides=2, 
        padding='SAME',
        data_format='NHWC' # Explicit, though default
    )

    print(f"Output contains NaN? {tf.reduce_any(tf.math.is_nan(y)).numpy()}")
    print(f"Output contains Inf? {tf.reduce_any(tf.math.is_inf(y)).numpy()}")
    print(f"Stats: min={tf.reduce_min(y).numpy()}, max={tf.reduce_max(y).numpy()}")

    if tf.reduce_any(tf.math.is_nan(y)):
        print("Detected NaNs in MaxPool output!")

if __name__ == "__main__":
    test_maxpool_nan()
```