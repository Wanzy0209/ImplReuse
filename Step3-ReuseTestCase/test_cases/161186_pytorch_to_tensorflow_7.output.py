import torch
import tensorflow as tf
import numpy as np
import sys

# Check for GPU availability to match the original CUDA context
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU found: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Memory tracking might not be accurate.")

# 1. Define Custom Autograd-like Function
# Mimicking torch.autograd.Function
@tf.custom_gradient
def MyOp(x):
    # Create large tensors to simulate memory load (2**20 floats = 4MB)
    out_0 = tf.zeros(2**20, dtype=tf.float32)
    out_1 = tf.zeros(2**20, dtype=tf.float32)

    # In PyTorch: ctx.save_for_backward(inp, out_0, out_1)
    # In TensorFlow, the grad function closure captures these variables.
    # We access them in the backward pass to ensure they are retained by the graph.
    def grad(dy):
        # Accessing "saved" tensors to mimic PyTorch's ctx.saved_tensors
        _ = out_0, out_1
        return None

    return out_0, grad

# 2. Define the operation function
def op_fn(x):
    return MyOp(x)

# 3. Setup Dataset
# Mimicking dummy_input = torch.nn.Parameter(...)
# We create a dataset that yields a tensor of similar size
dummy_input_shape = (2**20,)
dataset = tf.data.Dataset.from_tensors(tf.random.normal(dummy_input_shape))

# 4. Apply the Similar API: tf.compat.v1.distribute
# The provided snippet maps this to tf.data.experimental.service.distribute
# Note: This requires a running tf.data service cluster.
# We define the transformation here.

# To make the test case runnable without an external cluster, 
# we will attempt to apply it, but fall back to local iteration if it fails,
# while preserving the code structure for the API usage.

try:
    # Attempt to distribute the dataset
    # This is the direct adaptation of:
    # full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, ...)
    
    # We map the op first
    mapped_dataset = dataset.map(op_fn)
    
    # Apply distribute
    # Note: 'grpc://localhost:5000' is a placeholder. 
    # In a real environment, this would point to a dispatcher.
    distributed_dataset = tf.data.experimental.service.distribute(
        processing_mode="parallel_epochs",
        service="grpc://localhost:5000",
        job_name="memory_leak_test"
    )(mapped_dataset)
    
    # If successful, use this for iteration
    iter_dataset = distributed_dataset.repeat(1000)
    print("Using distributed dataset.")

except Exception as e:
    print(f"Could not connect to tf.data service (expected if service not running): {e}")
    print("Falling back to local dataset iteration to demonstrate memory logic.")
    # Fallback to local execution to ensure the script is runnable
    iter_dataset = dataset.map(op_fn).repeat(1000)

# 5. Run Loop and Check Memory
# Mimicking the for loop in the original bug report
print("Starting iterations...")
for i, data in enumerate(iter_dataset):
    if i >= 100: # Limit iterations for the demo
        break
        
    # In PyTorch: full_out.sum().backward()
    # Here we just consume the data. The custom gradient is defined but not executed
    # because we are not in a GradientTape context. However, the forward pass
    # (creation of out_0, out_1) happens.
    
    if i % 10 == 0:
        if gpus:
            mem_info = tf.config.experimental.get_memory_info(gpus[0].name)
            print(f"Iteration {i}, Memory Allocated: {mem_info['current'] / 1024**2:.2f} MiB")
        else:
            # Fallback for CPU
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                print(f"Iteration {i}, Memory RSS: {process.memory_info().rss / 1024**2:.2f} MiB")
            except ImportError:
                pass

print("Test complete.")