import torch
import tensorflow as tf
import numpy as np

def test_batch_parallel_grad():
    """
    Adapted test case for tf.compat.v1.tpu.batch_parallel based on the 
    PyTorch flex_attention backpropagation bug.
    
    The original bug involves a compiled function (torch.compile) calling an API (flex_attention)
    that takes a closure (score_mod) capturing a tensor (bias_mat) derived from an input (y).
    Backpropagation fails for y depending on graph breaks/compilation details.
    
    This test adapts the logic to TensorFlow's TPU batch_parallel API.
    We verify if gradients flow correctly through the compiled computation
    for inputs used to create intermediate tensors (bias_mat).
    """
    
    # Initialize TPU system (Required for batch_parallel)
    # Note: This code requires a TPU environment to execute fully.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU initialized.")
    except ValueError:
        print("TPU not found. This test case is designed for a TPU environment.")
        return

    # Define constants
    B, L, D = 2, 16, 64

    # Define inputs
    # In TF, we use tf.Variable or Tensors watched by GradientTape
    x = tf.random.normal((B, L, D))
    y = tf.random.normal((B, L))

    # Define the computation function to be parallelized
    # This corresponds to the `test` function in the PyTorch snippet
    def computation(x, y):
        # Materialize a bias matrix
        # PyTorch: bias_mat = y[b, q_idx] + y[b, kv_idx]
        # TF equivalent using broadcasting to create (B, L, L) matrix
        # y is (B, L) -> y[:, :, None] is (B, L, 1)
        # y[:, None, :] is (B, 1, L)
        bias_mat = y[:, :, tf.newaxis] + y[:, tf.newaxis, :]  # (B, L, L)

        # Simulate the attention mechanism logic
        # flex_attention(x_, x_, x_, score_mod=score_mod)
        # We perform a simplified attention operation where bias_mat is added to scores
        
        # Q, K, V setup
        q = x
        k = x
        v = x
        
        # Scores
        scores = tf.matmul(q, k, transpose_b=True)  # (B, L, L)
        
        # Apply score_mod logic (adding bias)
        # In the original bug, this happens inside a kernel passed a closure.
        # Here it happens inside the `computation` closure passed to batch_parallel.
        scores_mod = scores + bias_mat
        
        # Softmax and Output
        attn_weights = tf.nn.softmax(scores_mod)
        output = tf.matmul(attn_weights, v)
        
        return output

    # Use tf.GradientTape to track gradients
    with tf.GradientTape() as tape:
        # Watch the input tensors
        tape.watch(x)
        tape.watch(y)

        # Execute the compiled computation on TPU
        # tf.compat.v1.tpu.batch_parallel shards the computation along the batch dimension.
        # inputs is a list of arguments for the computation function.
        out = tf.compat.v1.tpu.batch_parallel(computation, [x, y], num_shards=1)
        
        # Calculate loss (mean of output)
        loss = tf.reduce_mean(out)

    # Calculate gradients
    grads = tape.gradient(loss, [x, y])
    grad_x, grad_y = grads

    # Check results
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"x grad is None: {grad_x is None}, Norm: {tf.norm(grad_x).numpy() if grad_x is not None else 0}")
    print(f"y grad is None: {grad_y is None}, Norm: {tf.norm(grad_y).numpy() if grad_y is not None else 0}")

    # Assertions corresponding to the original bug report
    # The bug was that y.grad was None or 0.
    assert grad_x is not None, "Gradient for x is None"
    assert tf.norm(grad_x) > 0, "Gradient for x is zero"
    
    assert grad_y is not None, "Gradient for y is None (Bug Reproduced)"
    assert tf.norm(grad_y) > 0, "Gradient for y is zero (Bug Reproduced)"
    
    print("Test Passed: Gradients propagated correctly.")

if __name__ == "__main__":
    test_batch_parallel_grad()