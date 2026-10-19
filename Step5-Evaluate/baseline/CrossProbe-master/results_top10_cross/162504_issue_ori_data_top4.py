```python
import tensorflow as tf

# Conversion: torch.cuda.manual_seed(42)
tf.random.set_seed(42)

# Conversion: torch.utils.checkpoint.checkpoint
# In TensorFlow, activation checkpointing is typically applied using the @tf.recompute_grad decorator
@tf.recompute_grad
def fn(x):
    # Conversion: torch.randn(1, device="cuda")
    # tf.random.normal generates random values from a normal distribution
    return x * tf.sigmoid(tf.random.normal(shape=(1,)))

# initialize device state
# Conversion: fn(torch.ones(1, device="cuda"))
fn(tf.ones((1,)))

# Eager execution
tf.random.set_seed(42)
# Conversion: torch.ones(1, device="cuda", requires_grad=True)
# tf.Variable is used for tensors that require gradients
eager_in = tf.Variable(tf.ones((1,)))

with tf.GradientTape() as tape:
    eager_out = fn(eager_in)

# Conversion: torch.autograd.grad(eager_out, eager_in)
eager_in_grad = tape.gradient(eager_out, eager_in)

# Graph Capture
# Conversion: torch.cuda.CUDAGraph() and with torch.cuda.graph(g):
# In TensorFlow, tf.function is used to capture a computation graph
@tf.function
def graph_execution():
    # Conversion: graph_in = torch.ones(1, device="cuda", requires_grad=True)
    # Creating a variable inside the graph function
    graph_in = tf.Variable(tf.ones((1,)))
    
    with tf.GradientTape() as tape:
        graph_out = fn(graph_in)
        
    # Conversion: graph_in_grad, = torch.autograd.grad(graph_out, graph_in)
    graph_in_grad = tape.gradient(graph_out, graph_in)
    return graph_in_grad

# Conversion: torch.cuda.manual_seed(42) and g.replay()
# Resetting seed before executing the captured graph to ensure RNG consistency
tf.random.set_seed(42)
graph_in_grad = graph_execution()

# Conversion: assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)
# Using assert_equal for exact match (rtol=0, atol=0)
tf.debugging.assert_equal(eager_in_grad, graph_in_grad, message="Mismatch in gradient outputs")

print(eager_in_grad)
print(graph_in_grad)
```