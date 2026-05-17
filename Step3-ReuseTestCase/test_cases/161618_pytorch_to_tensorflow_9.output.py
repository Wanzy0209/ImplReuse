import torch
import tensorflow as tf

# Reproduce the data setup from the original PyTorch bug report
m = 20120
k = 1536
n = 512

# Define the operation function similar to the PyTorch lambda
# PyTorch: torch.addmm(a, mat1, mat2) -> a + mat1 @ mat2
# TensorFlow: a + tf.matmul(mat1, mat2)
f = lambda a, mat1, mat2: a + tf.matmul(mat1, mat2)

# Check for GPU availability to match the .cuda() behavior in the original script
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device_name):
    # Initialize tensors similar to torch.randn
    a = tf.random.normal((m, n))
    mat1 = tf.random.normal((m, k))
    mat2 = tf.random.normal((k, n))

    # The target API: tf.name_scope
    # This replaces the inductor_config.patch context manager structurally
    # to group the operation execution.
    with tf.name_scope("addmm_autotune"):
        result = f(a, mat1, mat2)

    # Verify the result shape to ensure the operation executed correctly
    assert result.shape == (m, n), f"Expected shape ({m}, {n}), but got {result.shape}"
    
    print("Test executed successfully.")
    print(f"Result shape: {result.shape}")