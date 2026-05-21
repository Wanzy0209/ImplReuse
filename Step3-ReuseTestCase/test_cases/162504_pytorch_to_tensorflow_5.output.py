import torch
import tensorflow as tf

# Set seed for reproducibility
tf.random.set_seed(42)

# Define inputs for atrous_conv2d
# value shape: [batch, height, width, in_channels]
# filters shape: [filter_height, filter_width, in_channels, out_channels]
batch_size, height, width, in_channels = 1, 5, 5, 3
filter_height, filter_width, out_channels = 3, 3, 2
rate = 2

# Create tensors
value = tf.random.normal((batch_size, height, width, in_channels))
filters = tf.random.normal((filter_height, filter_width, in_channels, out_channels))

# 1. Eager Execution
with tf.GradientTape() as tape_eager:
    tape_eager.watch(value)
    eager_out = tf.nn.atrous_conv2d(value, filters, rate=rate, padding='SAME')
eager_grad = tape_eager.gradient(eager_out, value)

# 2. Graph Capture (using tf.function)
# In TensorFlow, tf.function captures the computation graph, analogous to torch.cuda.graph
@tf.function
def graph_fn(val, filt):
    with tf.GradientTape() as tape:
        tape.watch(val)
        out = tf.nn.atrous_conv2d(val, filt, rate=rate, padding='SAME')
    return out, tape.gradient(out, val)

graph_out, graph_grad = graph_fn(value, filters)

# 3. Assertions
# Check if outputs match
assert tf.reduce_all(tf.abs(eager_out - graph_out) < 1e-6).numpy(), "Mismatch in outputs"
# Check if gradients match
assert tf.reduce_all(tf.abs(eager_grad - graph_grad) < 1e-6).numpy(), "Mismatch in gradients"

print("Eager Output:\n", eager_out.numpy())
print("Graph Output:\n", graph_out.numpy())
print("Test Passed: Eager and Graph execution results match.")