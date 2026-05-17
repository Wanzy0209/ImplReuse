import torch
import tensorflow as tf
import numpy as np

def test_uint8_neg_add_with_range_producer():
    """
    Adapts the PyTorch bug reproduction logic (neg+add on uint8) 
    to use tf.compat.v1.train.range_input_producer as the data source.
    """
    # Disable eager execution to use tf.compat.v1 queues and sessions
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # 1. Setup range_input_producer
        # We use the producer to generate a base integer value.
        # To match the PyTorch bug where c=7, we will offset the producer's output.
        limit = 10
        producer = tf.compat.v1.train.range_input_producer(
            limit, num_epochs=1, shuffle=False, seed=0, capacity=32
        )
        
        # Dequeue a value from the producer
        base_val = producer.dequeue()

        # 2. Adapt the logic: Create a uint8 tensor 'c'
        # The original bug used c = torch.tensor(7, dtype=torch.uint8).
        # We cast the producer output to uint8 and add 7 to ensure we hit the specific 
        # arithmetic case (negation of 7) that triggered the bug.
        c = tf.cast(base_val, tf.uint8) + 7

        # 3. Define the float input 'x' matching the PyTorch test case
        x_np = np.array([[1.5410, -0.2934], [-2.1788, 0.5684]], dtype=np.float32)
        x = tf.constant(x_np)

        # 4. Perform the operations: c+x, neg(c), neg(c)+x
        # These operations correspond to the core logic of the reported bug.
        res0 = c + x
        res1 = tf.negative(c)
        res2 = tf.negative(c) + x

        # Initialize local variables (required for range_input_producer with num_epochs)
        # and global variables.
        sess.run([tf.compat.v1.local_variables_initializer(), 
                  tf.compat.v1.global_variables_initializer()])
        
        # Start queue runners to populate the queue
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Execute the graph
            r0, r1, r2, c_val = sess.run([res0, res1, res2, c])

            print(f"Input c (uint8): {c_val}")
            print(f"Input x: \n{x_np}")
            print(f"res[0] (c+x): \n{r0}")
            print(f"res[1] (neg(c)): {r1}")
            print(f"res[2] (neg(c)+x): \n{r2}")

            # Verification:
            # In the PyTorch bug, eager mode correctly handled uint8 wrapping (neg(7) -> 249),
            # while the compiled mode incorrectly treated it as float (-7).
            # TensorFlow should behave like PyTorch eager mode (preserving uint8 semantics).
            
            # neg(7) in uint8 is 249 (256 - 7)
            assert r1 == 249, f"Expected neg(7) to wrap to 249, got {r1}"
            
            # 249 + 1.5410 should be 250.5410
            expected_val = 249.0 + 1.5410
            assert np.isclose(r2[0, 0], expected_val), \
                f"Expected {expected_val}, got {r2[0, 0]}"
            
            print("\nTest passed: TensorFlow correctly handles uint8 negation/addition logic.")

        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_uint8_neg_add_with_range_producer()