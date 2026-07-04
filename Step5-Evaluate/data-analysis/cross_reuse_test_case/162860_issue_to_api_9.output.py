import tensorflow as tf

def inner(x):
    return x + 1

@tf.function
def fn(x):
    # The original issue requested more debug information (context) inside 
    # a compiled function. The similar API allows checking the execution context.
    is_eager = tf.compat.v1.executing_eagerly()
    
    # Reproduce the logic from the original bug report
    x = inner(x)
    return inner(x), is_eager

# Execute the function
result, is_eager_mode = fn(tf.ones(3))

# Assertions
# 1. Verify the calculation logic matches the original (1+1+1 = 3)
# We need to handle both Eager mode (where .numpy() works) and Graph mode (where it doesn't)
if tf.executing_eagerly():
    assert tf.reduce_all(result == 3).numpy()
else:
    with tf.compat.v1.Session() as sess:
        assert sess.run(tf.reduce_all(result == 3))

# 2. Verify the API behavior. By default, tf.function runs in graph mode,
# so executing_eagerly should return False inside the function.
# tf.compat.v1.executing_eagerly() returns a Python bool, so this works in both modes.
assert is_eager_mode == False