import torch
import tensorflow as tf
import numpy as np

# Check for GPU availability to ensure memory metrics are relevant
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("This test requires a GPU to run.")
    exit()

# Define a custom gradient operation (equivalent to torch.autograd.Function)
@tf.custom_gradient
def my_op(x):
    # Create large tensors similar to the PyTorch example
    # These tensors mimic the intermediate activations that might be saved for backward
    out_0 = tf.zeros(2**20, dtype=tf.float32)
    out_1 = tf.zeros(2**20, dtype=tf.float32)

    # The backward function
    def grad(d_out):
        # Access the tensors saved for backward (mimicking ctx.saved_tensors)
        # In TensorFlow, this is typically done via closure.
        # We perform a dummy operation to ensure they are part of the gradient graph.
        _ = out_0 * d_out
        _ = out_1 * d_out
        return None # No gradient for input x

    return out_0, grad

# Setup Distributed Strategy (equivalent context to torch.utils.checkpoint)
# We use MirroredStrategy to simulate a managed execution environment.
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create input variable
# We create it inside the strategy scope to ensure it is managed correctly
with strategy.scope():
    dummy_input = tf.Variable(tf.random.normal([2**20], dtype=tf.float32))

print("Starting loop to check for memory leaks...")

# Run the loop multiple times to detect memory growth
for i in range(100):
    with tf.GradientTape() as tape:
        # Execute the operation inside the strategy context
        # This mimics the checkpointed region in PyTorch
        def step_fn(x):
            return my_op(x)

        # strategy.run replicates the computation across devices
        distributed_out = strategy.run(step_fn, args=(dummy_input,))
        
        # Calculate loss (sum of outputs)
        # We reduce the distributed outputs to a single scalar for backward pass
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, distributed_out, axis=None)

    # Backward pass
    grads = tape.gradient(loss, dummy_input)
    
    # Explicitly delete gradients to free memory (similar to dummy_input.grad = None)
    del grads

    # Check memory usage
    # Note: In TF, memory management is eager. We check the allocator stats.
    # We look at the first GPU's memory.
    mem_info = tf.config.experimental.get_memory_info('GPU:0')
    current_mem_mb = mem_info['current'] / (1024**2)
    
    print(f"Iteration {i}: {current_mem_mb:.2f} MiB")

    # Optional: Assertion to fail if memory grows significantly beyond a threshold
    # This helps in automated testing environments
    if i > 10 and i % 10 == 0:
        # A simple heuristic: if memory grows by more than 100MB in 10 iterations, it might be a leak
        # (Adjust threshold based on baseline overhead)
        pass 

print("Test completed.")