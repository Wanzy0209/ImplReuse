import torch
import tensorflow as tf

# Define the eager execution function
def eager_fn(x, y):
    return x + y

# Define the compiled execution function (equivalent to torch.compile)
@tf.function
def compiled_fn(x, y):
    return x + y

# Setup inputs
# x is a non-empty complex tensor
x = tf.complex(tf.random.normal((1,)), tf.random.normal((1,)))
# y is a 0-dimensional (scalar) empty complex tensor
y = tf.empty((), dtype=tf.complex64)

# Test eager execution
try:
    result_eager = eager_fn(x, y)
    print("eager success")
except Exception as e:
    print(f"eager failed: {e}")

# Test compiled execution
try:
    result_compiled = compiled_fn(x, y)
    print("compiler success")
except Exception as e:
    print(f"compiler failed: {e}")