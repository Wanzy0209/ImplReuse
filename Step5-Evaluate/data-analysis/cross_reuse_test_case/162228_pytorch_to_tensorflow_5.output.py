import sys

try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    # Handle the specific GLIBCXX error mentioned in the traceback
    if "GLIBCXX" in str(e):
        print("Skipping test due to environment incompatibility:")
        print(f"  {e}")
        print("  The installed TensorFlow/Protobuf version requires a newer GLIBCXX (libstdc++) than what is available on the system.")
        sys.exit(0)
    else:
        raise

# Disable eager execution to use TF1 graph mode, which is the context for add_queue_runner
tf.compat.v1.disable_eager_execution()

# Define dimensions
B, L, D = 2, 16, 64

# Define variables x and y (equivalent to torch tensors with requires_grad=True)
x_var = tf.Variable(tf.random.normal([B, L, D]), name='x')
y_var = tf.Variable(tf.random.normal([B, L]), name='y')

# Setup a FIFOQueue to manage data flow
# This mimics the data passing in the compiled function
queue = tf.compat.v1.queue.FIFOQueue(
    capacity=10, 
    dtypes=[tf.float32, tf.float32], 
    shapes=[[B, L, D], [B, L]]
)

# Enqueue operation
enqueue_op = queue.enqueue([x_var, y_var])

# Create a QueueRunner
# The QueueRunner manages multiple threads that enqueue data asynchronously
qr = tf.compat.v1.train.QueueRunner(queue, [enqueue_op] * 2)

# --- Target API Usage ---
# Add the QueueRunner to the graph collection
# This is the API being tested for correct integration with the graph/gradients
tf.compat.v1.train.add_queue_runner(qr)
# -----------------------

# Dequeue the data to use in the computation graph
x, y = queue.dequeue()

# Replicate the logic from the PyTorch bug report:
# Materialize a bias matrix from y
# PyTorch: bias_mat = y[b, q_idx] + y[b, kv_idx]
# TensorFlow equivalent using broadcasting:
# y shape is (B, L). We want (B, L, L).
y_q = tf.expand_dims(y, axis=2)  # (B, L, 1)
y_k = tf.expand_dims(y, axis=1)  # (B, 1, L)
bias_mat = y_q + y_k             # (B, L, L)

# Dummy attention score calculation (mimicking flex_attention inputs)
# x is (B, L, D). We project to (B, L, L) scores.
# PyTorch: x_ = x[:, :, None].repeat(1, 1, 16, 1)
# We simplify this to a matrix multiplication to generate scores
scores = tf.matmul(x, x, transpose_b=True)  # (B, L, L)

# Apply the score_mod logic: score + bias
modified_scores = scores + bias_mat

# Calculate loss
loss = tf.reduce_mean(modified_scores)

# Compute gradients
# In TF1, we compute gradients with respect to the variables
grads = tf.gradients(loss, [x_var, y_var])

# Session execution
with tf.compat.v1.Session() as sess:
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Coordinator for managing queue threads
    coord = tf.train.Coordinator()
    
    # Start the queue runners (including the one added via add_queue_runner)
    threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)
    
    try:
        # Run the gradient computation
        gx_val, gy_val = sess.run(grads)
        
        # Verify gradients
        print(f"x: {(gx_val is not None) and (np.linalg.norm(gx_val) > 0)}, "
              f"y: {(gy_val is not None) and (np.linalg.norm(gy_val) > 0)}")
        
        assert gx_val is not None and np.linalg.norm(gx_val) > 0, "Gradient for x is None or zero"
        assert gy_val is not None and np.linalg.norm(gy_val) > 0, "Gradient for y is None or zero"
        print("Test passed: Gradients propagated correctly.")
        
    finally:
        # Stop the threads
        coord.request_stop()
        coord.join(threads)