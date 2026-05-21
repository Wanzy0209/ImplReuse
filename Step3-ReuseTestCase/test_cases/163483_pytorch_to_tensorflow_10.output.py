import torch
import tensorflow as tf
import numpy as np

# Enable numpy behavior for the experimental API
tf.experimental.numpy.enable_numpy_behavior()

# Setup: Create a 4D tensor
# PyTorch equivalent: torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
# In TensorFlow, the default memory layout for 4D tensors is NHWC (channels_last).
x = tf.reshape(tf.range(16, dtype=tf.float32), (2, 2, 2, 2))

# Capture input memory layout (strides)
x_arr = tf.experimental.numpy.asarray(x)
input_strides = x_arr.strides

# Operation: Apply cbrt
# PyTorch equivalent: torch.distributed.all_gather(x_list, x)
# Adapted to: tf.experimental.numpy.cbrt(x)
y = tf.experimental.numpy.cbrt(x)

# Capture output memory layout
y_arr = tf.experimental.numpy.asarray(y)
output_strides = y_arr.strides

# Verification
# 1. Check mathematical correctness (cbrt(x)^3 == x)
# 2. Check memory ordering preservation (strides match)
is_values_correct = np.allclose(y_arr ** 3, x_arr)
is_layout_preserved = (input_strides == output_strides)

print(f'Input strides: {input_strides}')
print(f'Output strides: {output_strides}')
print(f'Values correct: {is_values_correct}')
print(f'Memory ordering preserved: {is_layout_preserved}')

# The bug in PyTorch was that memory ordering was NOT preserved.
# We assert that it SHOULD be preserved for a correct implementation.
assert is_values_correct, "Mathematical values are incorrect."
assert is_layout_preserved, "Memory ordering was changed (Bug reproduced)."