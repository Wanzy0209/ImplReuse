import torch
import tensorflow as tf

MAX = 3
BATCH = 37

# Define inputs
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

def func(x, idxs):
    # Equivalent to x.square() * torch.nn.functional.one_hot(idxs, MAX)
    # tf.one_hot defaults to float32, casting to float64 to match x
    one_hot = tf.cast(tf.one_hot(idxs, MAX), tf.float64)
    return tf.square(x) * one_hot

def jacfunc(x, idxs):
    # Equivalent to torch.func.jacfwd(func, argnums=(0,))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = func(x, idxs)
    return tape.jacobian(y, x)

# 1. Test Eager (works in PyTorch)
print("Testing Eager execution...")
out = jacfunc(x, idxs)
assert out is not None
print("Eager execution passed.")

# 2. Test Compiled with name_scope
# Adapting to tf.keras.name_scope as the similar API context
# and tf.function as the compilation mechanism.

@tf.function
def compiled_jacfunc(x, idxs):
    with tf.keras.name_scope("dynamic_jacobian_scope"):
        return jacfunc(x, idxs)

print("Testing Compiled execution with name_scope...")
try:
    out = compiled_jacfunc(x, idxs)
    assert out is not None
    print("Compiled execution passed.")
except Exception as e:
    print(f"Error during compiled execution: {e}")
    raise