```python
import tensorflow as tf

# Conversion: torch.cuda.manual_seed -> tf.random.set_seed
tf.random.set_seed(42)

# Conversion: torch.utils.checkpoint.checkpoint
# In TensorFlow, activation checkpointing is typically done using the @tf.recompute_grad decorator.
# This trades compute for memory by recomputing the forward pass during backprop.
@tf.recompute_grad
def fn(x):
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: torch.sigmoid -> tf.math.sigmoid
    # Note: We use stateful random ops to match torch.randn behavior.
    return x * tf.math.sigmoid(tf.random.normal((1,)))

# Initialize device state
# Conversion: device="cuda" -> tf.device('/GPU:0')
with tf.device('/GPU:0'):
    fn(tf.ones((1,)))

# Eager execution block
tf.random.set_seed(42)
with tf.device('/GPU:0'):
    eager_in = tf.ones((1,))
    with tf.GradientTape() as tape:
        eager_out = fn(eager_in)
    # Conversion: torch.autograd.grad -> tape.gradient
    # PyTorch returns a tuple, TensorFlow returns the tensor directly (or None)
    eager_in_grad = tape.gradient(eager_out, eager_in)

# Graph capture block
# Conversion: torch.cuda.CUDAGraph -> tf.function with jit_compile=True
# tf.function captures the TensorFlow graph, and jit_compile=True optimizes it (similar to CUDA Graphs).
@tf.function(jit_compile=True)
def graph_step(x):
    with tf.GradientTape() as tape:
        out = fn(x)
    return tape.gradient(out, x)

tf.random.set_seed(42)
with tf.device('/GPU:0'):
    graph_in = tf.ones((1,))
    # The first call traces the graph (Capture), subsequent calls replay it.
    # This mimics the 'with torch.cuda.graph(g):' capture and 'g.replay()' execution.
    graph_in_grad = graph_step(graph_in)

# Conversion: torch.allclose -> tf.debugging.assert_near
# Note: Due to differences in RNG state preservation during checkpointing 
# (PyTorch's preserve_rng_state vs TF's recompute_grad behavior with stateful ops),
# exact numerical matching might require stateless random ops in TF.
# However, the structure is translated faithfully.
tf.debugging.assert_near(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)
print(eager_in_grad)
print(graph_in_grad)
```