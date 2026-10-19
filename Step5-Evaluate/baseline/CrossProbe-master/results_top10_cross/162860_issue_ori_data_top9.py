```python
import tensorflow as tf

def inner(x):
    return x + 1

# torch.compile is used to optimize the function. In TensorFlow, @tf.function
# is the equivalent decorator used to compile a Python function into a static graph.
@tf.function
def fn(x):
    x = inner(x)
    return inner(x)

fn(tf.ones(3))
```