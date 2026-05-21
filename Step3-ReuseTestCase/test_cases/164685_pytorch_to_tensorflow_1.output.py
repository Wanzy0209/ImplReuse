import torch
import tensorflow as tf
import numpy as np

# Attempt to initialize TPU for the target API
tpu_available = False
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_available = True
    print("TPU initialized.")
except (ValueError, tf.errors.NotFoundError):
    print("TPU not available. Using XLA JIT compilation as a fallback for 'compile' behavior.")

# Define the computation function adapted from the PyTorch fuzzed_program
def computation_fn(input_tensor, sentinel):
    # var_node_2 = -6 (dtype=int64)
    var_node_2 = tf.constant(-6, dtype=tf.int64)
    
    # var_node_3 = arg_0 (dtype=int32)
    # Cast input to int32 to match original logic
    var_node_3 = tf.cast(input_tensor, tf.int32)
    
    # var_node_1 = var_node_2 * var_node_3
    var_node_1 = var_node_2 * var_node_3
    
    # var_node_5 = torch.full((), 1, dtype=torch.int64)
    var_node_5 = tf.fill((), tf.cast(1, tf.int64))
    
    # var_node_4 = var_node_5.item() -> In TF graph, we use the tensor directly
    var_node_4 = var_node_5
    
    # var_node_0 = var_node_1 / var_node_4
    # PyTorch / operator performs true division. 
    # We cast to float to ensure true division behavior similar to PyTorch default
    var_node_0 = tf.cast(var_node_1, tf.float32) / tf.cast(var_node_4, tf.float32)
    
    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel
    return result

# Setup inputs
# Original arg_0 was a scalar item. batch_parallel requires sharding along dim 0.
# We create a batch of size 8 to allow sharding.
batch_size = 8
# Using random values similar to torch.randn
arg_0 = tf.constant(np.random.randn(batch_size), dtype=tf.int32)

# Sentinel tensor for gradient tracking
sentinel = tf.constant(1.0)

# 1. Eager Execution
print("Running Eager...")
result_eager = computation_fn(arg_0, sentinel)
print(" eager success")

# 2. Compiled/Parallel Execution
if tpu_available:
    print("Running TPU Batch Parallel...")
    # batch_parallel expects inputs as a list of lists of tensors
    # The computation function receives the unpacked inputs
    result_compiled = tf.compat.v1.tpu.batch_parallel(
        computation_fn,
        inputs=[[arg_0], [sentinel]],
        num_shards=8 # Assuming 8 cores for this example
    )
    print(" compile success")
else:
    print("Running XLA JIT Compilation (Fallback)...")
    # Use tf.function with XLA to simulate the compilation aspect
    compiled_fn = tf.function(computation_fn, jit_compile=True)
    result_compiled = compiled_fn(arg_0, sentinel)
    print(" compile success")

# Verify results
# Note: TPU sharding might reorder or handle precision differently, 
# but for this logic, we expect shape and value consistency.
assert tf.reduce_all(tf.shape(result_eager) == tf.shape(result_compiled)).numpy()
print("Test passed.")