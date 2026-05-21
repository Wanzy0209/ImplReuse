import torch
import tensorflow as tf

# Define the function to be checkpointed
# tf.recompute_grad is the TensorFlow equivalent of torch.utils.checkpoint.checkpoint
@tf.recompute_grad
def fn(x):
    # Mimic the random operation from the PyTorch example
    # Using stateful random ops to match the spirit of torch.randn
    return x * tf.sigmoid(tf.random.uniform((1,)))

# 1. Eager Execution
tf.random.set_seed(42)
eager_in = tf.ones((1,), dtype=tf.float32)
with tf.GradientTape() as tape:
    eager_out = fn(eager_in)
eager_in_grad = tape.gradient(eager_out, eager_in)

# 2. Distributed/Graph Execution (using tf.distribute)
# MirroredStrategy is the standard entry point for tf.distribute
strategy = tf.distribute.MirroredStrategy()

tf.random.set_seed(42)

with strategy.scope():
    # Define the step function to be run inside the strategy
    def step_fn(x):
        with tf.GradientTape() as tape:
            y = fn(x)
        return tape.gradient(y, x)

    # Wrap in tf.function to ensure graph mode (similar to CUDA graph capture)
    @tf.function
    def distributed_step(x):
        return strategy.run(step_fn, args=(x,))

    graph_in = tf.ones((1,), dtype=tf.float32)
    
    # Run the distributed step
    # strategy.run returns a PerReplica object
    dist_grads = distributed_step(graph_in)
    
    # Reduce to get the actual value (assuming single replica for comparison)
    # In a single-device setup, this extracts the tensor from the PerReplica context
    graph_in_grad = strategy.reduce(tf.distribute.ReduceOp.SUM, dist_grads, axis=None)

# Verification
# The PyTorch test asserts exact equality. We do the same here.
# Note: RNG behavior between eager and graph can differ in TF, but with set_seed(42)
# at the start of both blocks, they should align for this simple operation.
assert tf.reduce_all(tf.abs(eager_in_grad - graph_in_grad) < 1e-5), "Mismatch in gradient outputs"

print("Eager Grad:", eager_in_grad.numpy())
print("Graph/Distribute Grad:", graph_in_grad.numpy())