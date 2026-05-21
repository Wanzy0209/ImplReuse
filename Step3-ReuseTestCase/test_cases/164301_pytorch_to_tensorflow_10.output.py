import torch
import tensorflow as tf
import numpy as np

def test_range_input_producer_dim0_regression():
    """
    Adapts the PyTorch torch.compile regression test (dim0_mxfp8_floor)
    to tf.compat.v1.train.range_input_producer.

    Original Bug Context:
    - Dimensions: M=16384, K=16384, BLOCK_SIZE=32
    - Operation: Quantization along rows (dim0).
    - Issue: Performance regression when iterating/processing these rows.

    Adaptation Logic:
    - The 'dim0' operation implies iterating over the M dimension (16384 rows).
    - We configure range_input_producer to generate indices for these rows.
    - We verify the producer correctly handles the specific scale (limit=16384)
      and block size (batch size) mentioned in the bug report.
    """
    # Constants from the original bug report
    M = 16384
    K = 16384
    BLOCK_SIZE = 32

    # Disable eager execution to use v1 queue runners properly
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # Map the 'dim0' (row-wise) operation to the range limit.
        # We want to produce indices for each of the M rows.
        limit = M

        # Create the producer
        # shuffle=False to match the deterministic nature of the benchmark
        producer = tf.compat.v1.train.range_input_producer(
            limit=limit,
            shuffle=False,
            capacity=BLOCK_SIZE * 2,
            name="dim0_row_producer"
        )

        # Dequeue a batch corresponding to the BLOCK_SIZE
        # This mimics processing the matrix in blocks
        batch = producer.dequeue_many(BLOCK_SIZE)

        # Initialize variables
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            # Run the operation to get the first batch of indices
            result = sess.run(batch)

            # Verify the output matches the expected range [0, 1, ..., BLOCK_SIZE-1]
            expected = np.arange(BLOCK_SIZE, dtype=np.int32)
            assert np.array_equal(result, expected), \
                f"Regression detected: Expected indices {expected}, but got {result}"

            print("Test passed: range_input_producer correctly generated dim0 indices.")

        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_range_input_producer_dim0_regression()