import torch
import tensorflow as tf
import numpy as np

def test_string_input_producer_regression():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    
    This test adapts the logic from the torch.compile regression report:
    - It simulates a large workload (analogous to M=16384, K=16384 matrix).
    - It applies a specific buffer configuration (analogous to BLOCK_SIZE=32).
    - It verifies that the data pipeline processes the input correctly without 
      corruption or hanging, which would be the functional equivalent of a 
      performance regression or codegen failure in the input pipeline.
    """
    
    # Configuration adapted from the original bug report
    # Original: M 16384 K 16384 (Matrix dimensions)
    num_strings = 16384
    
    # Original: BLOCK_SIZE 32 (Quantization block size)
    # Adaptation: Queue capacity
    capacity = 32

    # Create a large tensor of strings to simulate the data workload
    string_tensor = tf.constant([f"file_{i}.dat" for i in range(num_strings)], dtype=tf.string)

    # Disable eager execution to ensure compat.v1 queue runners work as intended
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # Original: mode dim0_mxfp8_floor (Specific processing mode)
        # Adaptation: Use deterministic mode (shuffle=False) to verify exact output order
        queue = tf.compat.v1.train.string_input_producer(
            string_tensor,
            num_epochs=1,
            shuffle=False,
            capacity=capacity,
            name="string_producer_regression_test"
        )

        # Dequeue operation to retrieve data from the pipeline
        dequeue_op = queue.dequeue()

        # Initialize local variables (required for num_epochs counter)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start the queue runners (analogous to launching the kernel)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            # Verify the output by consuming the queue
            results = []
            for _ in range(num_strings):
                val = sess.run(dequeue_op)
                results.append(val.decode('utf-8'))

            # Assertion: Check if the pipeline processed all data correctly and in order
            expected = [f"file_{i}.dat" for i in range(num_strings)]
            assert results == expected, (
                f"Regression detected: Output mismatch. "
                f"Expected {len(expected)} items, got {len(results)}."
            )
            
            print("Test passed: string_input_producer handled the workload correctly.")

        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            # Clean up queue runners
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_string_input_producer_regression()