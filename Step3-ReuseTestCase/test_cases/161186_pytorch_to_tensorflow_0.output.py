import tensorflow as tf
import numpy as np

# Check for GPU availability to track memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configure memory growth to avoid allocating all memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
        device_name = gpus[0].name
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Memory tracking will be limited.")
    device_name = None

# Define the custom function (analogous to MyOp autograd Function)
# In the context of tf.data, this is the map_func passed to parallel_interleave.
def map_func(x):
    # Simulate the allocation of large tensors (out_0, out_1) in the original bug.
    # We create a dataset containing a large tensor to mimic the memory footprint
    # of the custom operation's outputs.
    large_tensor = tf.zeros((2**20,), dtype=tf.float32) # ~4 MB
    return tf.data.Dataset.from_tensors(large_tensor)

# Input dataset (analogous to dummy_input)
# We use a range to simulate multiple iterations/inputs being processed
input_dataset = tf.data.Dataset.range(1000)

# Apply the similar API: tf.data.experimental.parallel_interleave
# This wraps the map_func similarly to how checkpoint wrapped the op_fn.
# We use cycle_length=1 to mimic the sequential execution flow of the original test.
dataset = tf.data.experimental.parallel_interleave(
    input_dataset,
    map_func=map_func,
    cycle_length=1,
    sloppy=False
)

print("Starting iteration...")
# Loop analogous to the for i in range(1000) loop in the original bug report
for i, data in enumerate(dataset):
    # Consume the data (analogous to the backward pass triggering computation)
    _ = data.numpy()
    
    # Monitor memory usage
    if i % 100 == 0:
        if device_name:
            mem_info = tf.config.experimental.get_memory_info(device_name)
            # 'current' represents the current memory allocated by the device
            print(f"Iteration {i}, Memory Allocated: {mem_info['current'] / 1024**2:.2f} MiB")
        else:
            print(f"Iteration {i}")

print("Test completed.")