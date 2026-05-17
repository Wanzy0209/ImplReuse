import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

def foo(input):
    # Adapt torch.nn.functional.interpolate to tf.compat.v1.extract_image_patches
    # Note: These are different operations, but we adapt the test structure.
    # Input shape: (batch, height, width, channels)
    # Original PyTorch input: (1, 40, 1, 1) -> NCHW
    # TF input: (1, 1, 1, 40) -> NHWC
    
    patches = tf.compat.v1.extract_image_patches(
        images=input,
        ksizes=[1, 1, 1, 1],  # Size of the sliding window
        strides=[1, 1, 1, 1],  # Stride of the sliding window
        rates=[1, 1, 1, 1],    # Dilation
        padding='VALID'        # Padding
    )
    
    # Post-processing logic adapted from original
    # PyTorch: squeeze(0) -> removes batch dim
    # TF: squeeze(axis=0)
    squeeze = tf.squeeze(patches, axis=0)
    
    # PyTorch: argmin(1) -> argmin over channel dim (dim 1 in NCHW)
    # TF: argmin(axis=-1) -> argmin over depth (channel) dim in NHWC
    argmin = tf.argmin(squeeze, axis=-1)
    return argmin

# Generate input data
# Original: (1, 40, 1, 1)
# Adapted: (1, 1, 1, 40)
x = np.random.uniform(0, 10, size=(1, 1, 1, 40)).astype(np.float64)

# Eager execution
eager_res = foo(tf.constant(x))

# Compiled execution (tf.function is the TF equivalent of torch.compile)
cfoo = tf.function(foo)
compile_res = cfoo(tf.constant(x))

# Assertion
# Using numpy testing for assertion as in the original logic
np.testing.assert_allclose(eager_res.numpy(), compile_res.numpy())