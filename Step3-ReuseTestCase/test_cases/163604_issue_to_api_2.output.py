import torch
import tensorflow as tf
from tensorflow.keras import backend as K

# Leverage the similar API to set the data format convention.
# The PyTorch bug involves operations on tensors with specific strides/layouts.
# Setting the format to 'channels_first' aligns with PyTorch's default (N, C, L).
K.set_image_data_format('channels_first')

def foo(arg0, arg1):
    # t0 = arg0
    # PyTorch: size=(4, 503, 64, 504)
    # TensorFlow: shape=(4, 503, 64, 504)
    
    # t1 = t0.mean(dim=0)
    # PyTorch: size=(503, 64, 504)
    t1 = tf.reduce_mean(arg0, axis=0)
    
    # t2 = torch.nn.functional.relu(t1)
    t2 = tf.nn.relu(t1)
    
    # t3 = arg1
    # PyTorch: size=(5, 16, 1, 64)
    
    # t4 = t3.sum(dim=0)
    # PyTorch: size=(16, 1, 64)
    t4 = tf.reduce_sum(arg1, axis=0)
    
    # t5 = t4.transpose(2, 1)
    # PyTorch: size=(16, 64, 1)
    # TensorFlow transpose on 3D tensor (dims 0, 1, 2) -> swap 1 and 2
    t5 = tf.transpose(t4, [0, 2, 1])
    
    # t6 = torch.nn.functional.conv1d(t2, t5, stride=1, padding=0)
    # PyTorch Conv1d:
    #   Input: (N, C, L) = (503, 64, 504)
    #   Weight: (Out, In, K) = (16, 64, 1)
    # TensorFlow Conv1d (with data_format='NCW' for channels_first):
    #   Input: (Batch, InChannels, Length) = (503, 64, 504)
    #   Filter: (Kernel, InChannels, OutChannels) = (1, 64, 16)
    
    # Transpose t5 from (16, 64, 1) to (1, 64, 16) for TF filter shape
    filter_tf = tf.transpose(t5, [2, 1, 0])
    
    # Perform convolution. 
    # We explicitly use data_format='NCW' to match the global setting configured via the similar API.
    t6 = tf.nn.conv1d(
        input=t2,
        filters=filter_tf,
        stride=1,
        padding='VALID',
        data_format='NCW'
    )
    
    return t6

# Initialize inputs
# PyTorch: arg0 (4, 503, 64, 504), arg1 (5, 16, 1, 64)
arg0 = tf.random.normal((4, 503, 64, 504))
arg1 = tf.random.normal((5, 16, 1, 64))

# 1. Test Eager Execution
with tf.GradientTape() as tape_eager:
    out_eager = foo(arg0, arg1)
    loss_eager = tf.reduce_sum(out_eager)
grads_eager = tape_eager.gradient(loss_eager, [arg0, arg1])

print("Eager Execution Success! ")

# 2. Test Compiled Execution (tf.function) to mimic torch.compile
# This checks for Eager/Compile divergence as reported in the bug.
foo_compiled = tf.function(foo)

with tf.GradientTape() as tape_compiled:
    out_compiled = foo_compiled(arg0, arg1)
    loss_compiled = tf.reduce_sum(out_compiled)
grads_compiled = tape_compiled.gradient(loss_compiled, [arg0, arg1])

print("Compile Execution Success! ")

# 3. Assertions to check for divergence
# Check output
if not tf.reduce_all(tf.abs(out_eager - out_compiled) < 1e-5):
    raise AssertionError("Eager and Compiled outputs diverged!")

# Check gradients
for g_e, g_c in zip(grads_eager, grads_compiled):
    if g_e is not None and g_c is not None:
        if not tf.reduce_all(tf.abs(g_e - g_c) < 1e-5):
            raise AssertionError("Eager and Compiled gradients diverged!")
    elif g_e is not g_c:
        raise AssertionError("Gradient structure mismatch (None vs Tensor)")

print("Test Passed: No divergence between Eager and Compiled modes.")