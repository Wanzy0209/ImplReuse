import torch
import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor
import numpy as np

# Setup Mesh
# Using default device to ensure the test runs in various environments (CPU/GPU)
mesh = dtensor.create_mesh([("x", 1)], devices=dtensor.default_device())

# Define Layout (Replicated for simplicity, analogous to single device behavior)
layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)

def f(x, y):
    # PyTorch: x.copy_(x.flip(1))
    # TF equivalent: In-place assignment of reversed tensor
    # Note: x must be a tf.Variable (or dtensor.DVariable) to support assign
    x.assign(tf.reverse(x, axis=[1]))

    # PyTorch: y = y.sum(dim=1, keepdim=True) + y
    # TF equivalent
    y = tf.reduce_sum(y, axis=1, keepdims=True) + y

    return x + y

# Create input data
# Keeping dimensions similar to the original bug report
x_val = tf.random.normal((20, 1024 * 1024))
y_val = tf.random.normal((20, 1024 * 1024))

# --- Reference Execution (Local Eager) ---
x_ref = tf.Variable(x_val)
y_ref = tf.Variable(y_val)
ref_result = f(x_ref, y_ref)

# --- Target Execution (DTensor via copy_to_mesh) ---
# Use copy_to_mesh to distribute the tensors onto the mesh
x_dt_tensor = dtensor.copy_to_mesh(x_val, layout)
y_dt_tensor = dtensor.copy_to_mesh(y_val, layout)

# Wrap in DVariable to support in-place operations (assign) required by the logic
x_dt = dtensor.DVariable(x_dt_tensor)
y_dt = dtensor.DVariable(y_dt_tensor)

act_result = f(x_dt, y_dt)

# --- Verification ---
# Convert DTensor result back to local tensor for comparison
act_result_local = dtensor.array_to_tensor(act_result)

# Assert numerical closeness
# This mimics torch.testing.assert_close
tf.debugging.assert_near(ref_result, act_result_local, message="DTensor result differs from reference")

print("Test passed.")