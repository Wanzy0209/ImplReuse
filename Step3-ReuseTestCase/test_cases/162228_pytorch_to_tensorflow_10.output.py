import tensorflow as tf

# Disable eager execution to properly use tf.compat.v1 queues and sessions
tf.compat.v1.disable_eager_execution()

def test_range_input_producer_gradients():
    """
    Adapted test case for tf.compat.v1.train.range_input_producer.
    Verifies that gradients flow correctly to variables that depend on the 
    output of the range_input_producer, similar to the original PyTorch bug 
    where gradients failed to flow to 'y' inside a compiled context.
    """
    # Define a variable 'y' analogous to the tensor 'y' in the original bug report.
    # Shape [B, L] -> [2, 16]
    y = tf.Variable(tf.random.normal([2, 16], dtype=tf.float32), name='y')

    # Setup range_input_producer
    # limit=16 matches L in the original test case.
    # This acts as the source of indices, analogous to the arange/index generation 
    # in the original PyTorch code.
    producer = tf.compat.v1.train.range_input_producer(
        limit=16, 
        num_epochs=None, 
        shuffle=False, 
        capacity=32
    )

    # Dequeue a batch of indices (L=16)
    # This mimics the index generation (q_idx, kv_idx) in the original snippet.
    indices = producer.dequeue_many(16)
    indices = tf.cast(indices, tf.int32)

    # Mimic the logic: bias_mat = y[b, q_idx] + ...
    # We gather from y using the produced indices.
    # Let's assume we are gathering from the first batch element (b=0).
    # y[0] has shape [16], indices has shape [16].
    gathered_values = tf.gather(y[0], indices)

    # Define a loss (analogous to .mean().backward() in the original snippet).
    # In the original, flex_attention output was reduced to a scalar.
    loss = tf.reduce_mean(gathered_values)

    # Compute gradients
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=[y])

    # Session execution
    with tf.compat.v1.Session() as sess:
        # Initialize variables (including local variables for the queue epoch counter)
        sess.run([tf.compat.v1.global_variables_initializer(), 
                  tf.compat.v1.local_variables_initializer()])

        # Start queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Run the gradient computation
            # grads_and_vars is a list of (gradient, variable) tuples
            grad_tensor, var_tensor = grads_and_vars[0]
            grad_val = sess.run(grad_tensor)

            # Verify gradients
            # In the original bug, y.grad was None or 0.
            # Here we assert it is not None and has a norm > 0.
            assert grad_val is not None, "Gradient for y is None"
            grad_norm = tf.linalg.norm(grad_val).eval(session=sess)
            assert grad_norm > 0, f"Gradient for y is zero (norm: {grad_norm})"
            
            print("Test passed: Gradients flowed correctly to y.")

        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_range_input_producer_gradients()