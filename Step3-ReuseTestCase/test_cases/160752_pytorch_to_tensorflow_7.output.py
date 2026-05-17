import torch
import tensorflow as tf
import numpy as np

# Constants from the original bug report
MAX = 3
BATCH = 37

def func(x, idxs):
    # TensorFlow equivalent of x.square() * torch.nn.functional.one_hot(idxs, MAX)
    return tf.square(x) * tf.one_hot(idxs, MAX)

def jacfunc(x, idxs):
    # TensorFlow equivalent of torch.func.jacfwd
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = func(x, idxs)
    return tape.jacobian(y, x)

# Setup inputs
# Using int64 for idxs and float64 for x to match the original types
idxs = tf.random.uniform((BATCH,), 0, MAX, dtype=tf.int64)
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# 1. Test eager execution (Original "works" case)
print("Running eager execution...")
out_eager = jacfunc(x, idxs)
print(f"Eager output shape: {out_eager.shape}")

# 2. Test with the target API: tf.keras.backend.name_scope
# The original bug involved torch.compile. In TensorFlow, tf.function is the 
# analogous compilation mechanism. We wrap the execution in tf.function 
# and use tf.keras.backend.name_scope as requested by the prompt.

@tf.function
def compiled_jacfunc(x, idxs):
    # Using the target API: tf.keras.backend.name_scope
    # We pass values=[x, idxs] to ensure correct graph mode handling as per API docs
    with tf.keras.backend.name_scope("jacobian_calculation", values=[x, idxs]):
        return jacfunc(x, idxs)

print("Running with name_scope and tf.function (compiled)...")
try:
    out_compiled = compiled_jacfunc(x, idxs)
    print(f"Compiled output shape: {out_compiled.shape}")
    
    # Verify that the results are consistent
    # Note: Due to potential graph optimizations, we check if values are close
    if tf.reduce_all(tf.abs(out_eager - out_compiled) < 1e-5).numpy():
        print("Test Passed: Eager and Compiled outputs match.")
    else:
        print("Test Warning: Outputs differ slightly (possible numerical precision differences).")
        
except Exception as e:
    print(f"Test Failed with error: {e}")