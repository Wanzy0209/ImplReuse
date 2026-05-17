import tensorflow as tf
import sys

# Check for GPU availability to perform memory checks
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("This test requires a GPU to check for memory leaks. Exiting.")
    sys.exit(0)

# Configure memory growth to prevent TensorFlow from allocating all memory at once
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# Define a custom gradient function to mimic the PyTorch autograd.Function
# This wraps the target API: tf.nn.atrous_conv2d
@tf.custom_gradient
def my_custom_op(inp):
    # Create a filter for atrous convolution
    # Shape: [filter_height, filter_width, in_channels, out_channels]
    filter_shape = [3, 3, 1, 1]
    filters = tf.constant(tf.random.normal(filter_shape), dtype=tf.float32)
    
    # Perform atrous convolution (the target API)
    # Mimicking the logic of saving outputs for backward
    out = tf.nn.atrous_conv2d(inp, filters, rate=2, padding='SAME')
    
    # Create a secondary tensor to save, mimicking the PyTorch bug's context saving
    out_1 = tf.zeros_like(out)
    
    # Save tensors for backward (captured in closure)
    saved_tensors = [inp, out, out_1]

    def grad(d_out):
        # Access saved tensors to ensure they are retained
        _ = saved_tensors
        # Return gradient for input (None for filter)
        return tf.zeros_like(inp)

    return out, grad

# Setup dummy input
# PyTorch used 2**20 elements. We use a similar size tensor [1, 1024, 1024, 1]
dummy_input = tf.Variable(tf.random.normal([1, 1024, 1024, 1]))

print("Starting memory leak test loop...")

initial_mem = 0
for i in range(100):
    with tf.GradientTape() as tape:
        # Call the custom op wrapping the target API
        # Note: PyTorch bug involved use_reentrant=False in checkpoint.
        # TF does not have an exact equivalent flag for atrous_conv2d, 
        # but we test the interaction of the op with the autograd graph.
        full_out = my_custom_op(dummy_input)
        loss = tf.reduce_sum(full_out)
    
    # Backward pass
    grads = tape.gradient(loss, dummy_input)
    
    # Check memory
    mem_info = tf.config.experimental.get_memory_info('GPU:0')
    current_mem_mb = mem_info['current'] / (1024 ** 2)
    
    if i == 0:
        initial_mem = current_mem_mb

    print(f"Iteration {i}: {current_mem_mb:.2f} MiB")
    
    # Ensure gradients are computed
    assert grads is not None

# Verify memory behavior
final_mem = current_mem_mb
mem_growth = final_mem - initial_mem

print(f"\nInitial Memory: {initial_mem:.2f} MiB")
print(f"Final Memory: {final_mem:.2f} MiB")
print(f"Memory Growth: {mem_growth:.2f} MiB")

# In the original PyTorch bug, memory grew unbounded (leaked).
# We assert that memory growth is minimal (allowing for some fragmentation/overhead).
# If this assertion fails, it indicates a similar memory leak.
assert mem_growth < 50.0, f"Memory leak detected! Growth: {mem_growth:.2f} MiB"

print("Test passed. No significant memory leak detected.")