import torch
import tensorflow as tf
from tensorflow.experimental import dtensor

# Setup a minimal mesh for DTensor execution
# This creates a single-device mesh to ensure the test is runnable locally
mesh = dtensor.create_mesh([("x", [1])], devices=["CPU:0"])
layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

def foo(x):
    # Define the constant tensor with uint8, matching the bug report
    c = tf.constant(7, dtype=tf.uint8)
    return c + x, tf.negative(c), tf.negative(c) + x

# Input data matching the PyTorch seed (torch.manual_seed(0))
x = tf.constant([[1.5410, -0.2934], [-2.1788, 0.5684]], dtype=tf.float32)
print(f"input: {x}")

# 1. Eager Execution (Baseline)
res = foo(x)

# 2. DTensor Execution using copy_to_mesh
# We copy the input tensors to the mesh to test the API behavior
# Note: We explicitly copy c as well to ensure the arithmetic happens on DTensors
c = tf.constant(7, dtype=tf.uint8)
x_dt = dtensor.copy_to_mesh(x, layout)
c_dt = dtensor.copy_to_mesh(c, layout)

# Perform operations on the DTensors
cres = (c_dt + x_dt, tf.negative(c_dt), tf.negative(c_dt) + x_dt)

# Print results for comparison
print(f"res[0]: {res[0]}")
print(f"cres[0]: {cres[0]}")
'''
Expected: res[0] and cres[0] should match (8.5410, 6.7066, etc.)
'''

print(f"res[1]: {res[1]}")
print(f"cres[1]: {cres[1]}")
'''
Expected: In TensorFlow, neg(uint8) promotes to int32, so -7.
This differs from PyTorch eager (which wraps to 249), but matches PyTorch's buggy compiled output.
We verify that copy_to_mesh maintains TensorFlow's standard behavior.
'''

print(f"res[2]: {res[2]}")
print(f"cres[2]: {cres[2]}")
'''
Expected: -7 + x -> [[-5.4590, -7.2934], [-9.1788, -6.4316]]
'''

# Assertions to verify correctness
# Verify that the DTensor results match the Eager results
tf.debugging.assert_near(res[0], cres[0])
tf.debugging.assert_near(res[1], cres[1])
tf.debugging.assert_near(res[2], cres[2])

print("Test passed: DTensor copy_to_mesh preserves arithmetic semantics for uint8.")