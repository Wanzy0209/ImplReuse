import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Adapt the PyTorch test case to TensorFlow
# Original PyTorch input: torch.randn(4, 6, 7) -> (Channels, Height, Width)
# TensorFlow input (NHWC): (Batch, Height, Width, Channels)
# We add a batch dimension of 1.
x = tf.random.normal((1, 6, 7, 4))

# Original PyTorch parameters:
# kernel_size=[1, 6], stride=[4, 9], ceil_mode=True, divisor_override=3
#
# Mapping to tf.compat.v1.nn.depthwise_conv2d_native:
# 1. kernel_size=[1, 6] -> filter shape [filter_height, filter_width, in_channels, channel_multiplier]
#    We use channel_multiplier=1 to maintain the same number of channels.
# 2. stride=[4, 9] -> strides=[1, 4, 9, 1] (batch, height, width, channels)
# 3. ceil_mode=True -> padding='SAME' (This mimics the output size calculation behavior of ceil_mode in this context)
# 4. divisor_override=3 -> In convolution, we simulate averaging by setting filter weights.
#    AvgPool sums elements and divides by divisor_override.
#    Convolution sums elements * weights.
#    To mimic division by 3, we set weights to 1/3.0.

filter_shape = [1, 6, 4, 1] # [H, W, in_C, out_C_multiplier]
# Initialize filter with 1/3.0 to mimic divisor_override=3
filter_weights = tf.ones(filter_shape) / 3.0

strides = [1, 4, 9, 1]
padding = 'SAME' 

# Run the operation
try:
    out = tf.compat.v1.nn.depthwise_conv2d_native(
        input=x,
        filter=filter_weights,
        strides=strides,
        padding=padding,
        data_format='NHWC'
    )

    print("TensorFlow Output Shape:", out.shape)
    print("TensorFlow Output:")
    print(out)

    # Basic assertion to ensure the operation runs and produces a tensor
    assert out.shape == (1, 2, 1, 4), f"Expected shape (1, 2, 1, 4), got {out.shape}"
    print("\nTest passed: Operation executed successfully with expected shape.")

except Exception as e:
    print(f"Test failed with error: {e}")