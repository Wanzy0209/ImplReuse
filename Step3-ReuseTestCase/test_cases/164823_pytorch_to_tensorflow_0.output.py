import torch
import tensorflow as tf

def computation_fn(x):
    """
    TensorFlow equivalent of the PyTorch model logic:
    1. Convert dense tensor to sparse.
    2. Perform arithmetic operation.
    3. Convert back to dense.
    """
    # PyTorch: x.to_sparse()
    x_sparse = tf.sparse.from_dense(x)
    
    # PyTorch: x_sparse * 2
    # TensorFlow uses tf.sparse.multiply for element-wise operations on sparse tensors
    result_sparse = tf.sparse.multiply(x_sparse, 2)
    
    # PyTorch: result.to_dense()
    return tf.sparse.to_dense(result_sparse)

# Create input data
# PyTorch: torch.randn(10, 10)
x = tf.random.normal((10, 10))

# 1. Run in Eager mode
print("Eager output:", computation_fn(x))

# 2. Run using the specific API: tf.compat.v1.tpu.rewrite
# This attempts to compile the function for TPU execution (XLA).
# Note: This specific API is designed for TPU contexts.
try:
    # The API expects the computation function and a list of inputs
    compiled_output = tf.compat.v1.tpu.rewrite(computation_fn, inputs=[x])
    print("Compiled output:", compiled_output)
except Exception as e:
    # Handling potential environment issues (e.g., no TPU initialized) 
    # to ensure the script is runnable for inspection, while demonstrating the API call.
    print(f"Exception during TPU rewrite (expected if no TPU present): {e}")