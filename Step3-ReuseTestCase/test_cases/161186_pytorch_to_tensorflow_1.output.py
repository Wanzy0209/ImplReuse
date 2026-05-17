import torch
import tensorflow as tf
import sys

# Enable eager execution to mimic PyTorch's dynamic graph behavior
tf.compat.v1.enable_eager_execution()

# Check for GPU availability for memory reporting
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the custom gradient function equivalent to torch.autograd.Function
@tf.custom_gradient
def MyOp(inp: tf.Tensor):
    # Forward pass: Create large tensors
    out_0 = tf.zeros(2**20, dtype=tf.float32)
    out_1 = tf.zeros(2**20, dtype=tf.float32)

    # Define the backward pass
    def grad(dA, dB):
        # Access the tensors saved in the forward pass (captured by closure)
        # This is equivalent to ctx.saved_tensors in PyTorch
        _ = out_0 * 1.0
        _ = out_1 * 1.0
        return None # No gradient for input

    return out_0, out_1, grad

# Create a dummy input variable
dummy_input = tf.Variable(tf.random.normal([2**20], dtype=tf.float32))

# Loop to check for memory leaks
print("Starting memory leak test...")
for i in range(1000):
    with tf.GradientTape() as tape:
        # Define the loop body for while_loop
        # We need to accumulate the output to ensure gradients are calculated
        def body_fn(idx, acc):
            # Apply the custom op
            o0, o1 = MyOp(dummy_input)
            # Return one of the outputs to accumulate
            return idx + 1, acc + o0

        def cond_fn(idx, acc):
            return idx < 10 # Run the loop 10 times

        # Execute the while loop (similar to torch.utils.checkpoint.checkpoint)
        _, full_out = tf.compat.v1.while_loop(cond_fn, body_fn, [tf.constant(0), tf.constant(0.0)])
        
        loss = tf.reduce_sum(full_out)

    # Calculate gradients (backward pass)
    _ = tape.gradient(loss, dummy_input)

    # Report memory
    if gpus:
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"{i} {mem_info['current'] / 1024**2:.2f} MiB")
        except Exception as e:
            print(f"{i} Memory info unavailable: {e}")
    else:
        if i % 100 == 0:
            print(f"{i} (No GPU available for memory check)")

print("Test completed.")