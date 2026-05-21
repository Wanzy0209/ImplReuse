import torch
import tensorflow as tf
import numpy as np

def test_range_input_producer_compilation():
    """
    Adapted test case for tf.compat.v1.train.range_input_producer based on 
    PyTorch issue 161969 regarding torch.compile and contiguity.
    
    This test verifies the behavior of the TensorFlow API when used within
    a compiled graph context (tf.function/graph mode) performing linear algebra
    operations similar to the original bug report.
    """
    
    # Disable eager execution to use tf.compat.v1 queues and simulate graph compilation
    tf.compat.v1.disable_eager_execution()

    def logp(x, matrix):
        # Mimic the original logic: Cholesky, Inverse, Matmul
        # Note: In TensorFlow, tensors are generally row-major, but we perform
        # the same operations to check for consistency.
        
        # Original: p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
        p_mat_sqrt = tf.linalg.cholesky(matrix)
        
        # Original: # print(matrix.is_contiguous()) # Uncomment this line to make the code run
        # In TF, tf.print is the graph-aware equivalent. We leave it commented to 
        # mirror the original bug report's structure.
        # tf.print("Matrix contiguity check placeholder")

        # Original: p_mat_sqrt_inv = p_mat_sqrt.inverse()
        p_mat_sqrt_inv = tf.linalg.inv(p_mat_sqrt)
        
        # Original: val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
        # x is expected to be (5, 3) inside the map, so x[0, :] is (3,)
        x_slice = x[0, :]
        
        # Matmul: (3, 3) @ (3,) -> (3,)
        mul = tf.linalg.matmul(p_mat_sqrt_inv, tf.expand_dims(x_slice, axis=1))
        
        # Square and Sum
        val = tf.reduce_sum(tf.square(mul))
        
        return -val / 2

    # Wrap in tf.function to mimic torch.compile behavior
    compiled_logp = tf.function(logp)

    # Setup the graph and session
    with tf.compat.v1.Session() as sess:
        # 1. Use the Similar API: tf.compat.v1.train.range_input_producer
        # to generate input data indices.
        # Original data shape: (2, 5, 3). We generate 2 indices.
        producer = tf.compat.v1.train.range_input_producer(limit=2, shuffle=False)
        indices = producer.dequeue_many(2) # Shape: (2,)

        # Create the actual data tensor `x` based on the producer output
        # Original: data = torch.zeros((2, 5, 3))
        data_pool = tf.zeros((2, 5, 3), dtype=tf.float32)
        x_batch = tf.gather(data_pool, indices) # Shape: (2, 5, 3)

        # 2. Define matrix `p`
        # Original: p = torch.diag(torch.tensor((20., 0.5, 5,))**2)
        p_vals = tf.constant([20., 0.5, 5.], dtype=tf.float32)
        p = tf.linalg.diag(tf.square(p_vals))

        # 3. Mimic torch.vmap(torch.func.grad(logp, 0), (0, None))
        # We use tf.map_fn to iterate over the batch (vmap equivalent)
        # and tf.GradientTape for the gradient.
        
        def compute_grad(elem):
            with tf.GradientTape() as tape:
                tape.watch(elem)
                loss = compiled_logp(elem, p)
            return tape.gradient(loss, elem)

        # Apply the map (vmap)
        score_func = tf.map_fn(compute_grad, x_batch, dtype=tf.float32)

        # 4. Execution
        # Initialize variables (required for range_input_producer epochs counter)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        
        # Start queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            # Run the compiled function
            result = sess.run(score_func)
            
            # Basic assertion to ensure execution completed and returned valid shape
            assert result.shape == (2, 5, 3), f"Expected shape (2, 5, 3), got {result.shape}"
            print("Test passed. Result shape:", result.shape)
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_range_input_producer_compilation()