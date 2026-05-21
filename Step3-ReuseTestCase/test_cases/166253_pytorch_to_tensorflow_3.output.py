import torch
import tensorflow as tf

# Define the function using the similar API: tf.compat.v1.convert_to_tensor
# We mimic torch.full((2,), x) by converting a list [x, x] to a tensor.
def func_nojit(x):
    return tf.compat.v1.convert_to_tensor([x, x], dtype=tf.float64)

# Apply TensorFlow compilation (equivalent to torch.compile)
func_jit = tf.function(func_nojit)

# Test values
x1 = 5.0
x2 = 10.0

print("Testing func_nojit:")
res1 = func_nojit(x1)
res2 = func_nojit(x2)
print(res1)
print(res2)

print("\nTesting func_jit (tf.function):")
res1_jit = func_jit(x1)
res2_jit = func_jit(x2)
print(res1_jit)
print(res2_jit)

# Assertions to verify correct behavior
# The original PyTorch bug would cause the second compiled call to return [5., 5.]
assert res1.numpy().tolist() == [5.0, 5.0]
assert res2.numpy().tolist() == [10.0, 10.0]
assert res1_jit.numpy().tolist() == [5.0, 5.0]
assert res2_jit.numpy().tolist() == [10.0, 10.0], "Bug reproduced: Compiled function returned cached value"