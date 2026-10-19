```python
import tensorflow as tf

keys = range(10)
allowed = [0, 1, 2, 3]

# torch.compile is roughly equivalent to tf.function in TensorFlow
@tf.function
def fn(x):
    x = x + 1

    # torch._dynamo.graph_break() forces the compiler to stop tracing and execute eagerly.
    # In TensorFlow, tf.py_function is used to run arbitrary Python code inside a graph,
    # effectively acting as a graph break.
    def graph_break_block(x_val):
        # This block runs in eager mode
        key = [key for key in keys if key in allowed]

        def inner():
            nonlocal key

        return x_val + key[0]

    return tf.py_function(graph_break_block, inp=[x], Tout=tf.float32)

# torch.ones(3) -> tf.ones(3)
fn(tf.ones(3))
```