import torch
import tensorflow as tf
import numpy as np

# Replicating the setup from the PyTorch issue
B, H, W, C = 20, 2, 2, 128

# Setup input and layer
x = tf.Variable(tf.random.normal((B, H, W, C)))
linear = tf.keras.layers.Dense(C, use_bias=False)

# Forward pass
values = linear(x)
# Reshape (equivalent to view in PyTorch)
values_view = tf.reshape(values, (B, H * W, C))

# Print forward strides
# Note: TF tensors don't have a .stride() method directly, so we check via numpy
print("forward strides", values_view.shape, values_view.numpy().strides)

# Define weights for the operation
weights = tf.random.normal((B, H * W, C))

# Operation: tf.keras.ops.hstack
# Original PyTorch code used einsum, here we adapt to use the similar API hstack
with tf.GradientTape() as tape:
    # hstack concatenates tensors along the second axis (horizontally)
    # Input shapes: (B, H*W, C) and (B, H*W, C)
    # Result shape: (B, H*W, 2*C)
    result = tf.keras.ops.hstack([values_view, weights])

# Backward pass
# Mimicking result.backward(torch.ones_like(result))
# which computes the gradient of the sum of the result w.r.t the inputs
grads = tape.gradient(tf.reduce_sum(result), values_view)

# Check gradient strides
print("values_view grad - shape:", grads.shape, ", stride:", grads.numpy().strides)

# Assertion to check if strides match (preserving the core logic of the bug report)
# In the original bug, these strides differed.
if values_view.numpy().strides != grads.numpy().strides:
    print("ISSUE DETECTED: Forward and backward strides do not match.")
else:
    print("SUCCESS: Forward and backward strides match.")