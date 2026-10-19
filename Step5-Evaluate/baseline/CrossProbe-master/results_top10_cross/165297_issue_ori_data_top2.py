```python
import tensorflow as tf

# TensorFlow uses NHWC (channels_last) by default for performance on GPU.
# We explicitly place operations on GPU to match torch.device("cuda").
# Note: TF tensors are immutable, so "contiguous" checks are not applicable in the same way.
device = '/GPU:0'

def test_maxpool_nan():
    # Large input tensor
    N, C, H, W = 84, 64, 512, 960
    
    # Use a device context to place tensors on the GPU
    with tf.device(device):
        # Case 1: bfloat16 + channels_last => NaN
        # In TensorFlow, the default data format for 4D tensors is NHWC (channels_last).
        # We create the tensor directly in this format.
        x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

        # Case 2: float32 + channels_last => illegal memory access
        # x = tf.random.normal((N, H, W, C), dtype=tf.float32)  # default float32

        # Convert to NHWC channels_last layout
        # PyTorch: x = x.to(memory_format=torch.channels_last)
        # TensorFlow: Tensors are created in NHWC format by default. 
        # No conversion needed, but we ensure it is a tf.Tensor.
        x = tf.identity(x) 

        # Uncommenting the line below avoids NaNs
        # x = x.contiguous()
        # TensorFlow tensors are immutable and effectively always contiguous.

        # PyTorch: print(f"Input tensor: contiguous={x.is_contiguous()}, channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
        # TensorFlow does not expose is_contiguous() checks.
        print(f"Input tensor: shape={x.shape}, dtype={x.dtype}")

        # PyTorch: print(f"Input stride: {x.stride()}")
        # TensorFlow does not expose stride() in the public API for debugging.
        # print(f"Input stride: {x.stride()}")

        # PyTorch: pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        # TensorFlow: padding='same' approximates the behavior of padding=1 for odd kernels.
        pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        y = pool(x)

        # PyTorch: torch.isnan(y).any().item()
        has_nan = tf.reduce_any(tf.math.is_nan(y))
        print(f"Output contains NaN? {has_nan.numpy()}")
        
        # PyTorch: torch.isinf(y).any().item()
        has_inf = tf.reduce_any(tf.math.is_inf(y))
        print(f"Output contains Inf? {has_inf.numpy()}")
        
        # PyTorch: y.min().item(), y.max().item()
        print(f"Stats: min={tf.reduce_min(y).numpy()}, max={tf.reduce_max(y).numpy()}")

        if has_nan:
            print("Detected NaNs in MaxPool output!")

if __name__ == "__main__":
    test_maxpool_nan()
```