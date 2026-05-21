import torch
import tensorflow as tf
import numpy as np

# Ensure we are in graph mode (similar to the effect of torch.compile)
tf.compat.v1.disable_eager_execution()

def test_range_input_producer_with_pipeline_schedule():
    """
    Test case adapted from PyTorch Issue 161904.
    Verifies that tf.compat.v1.train.range_input_producer works correctly
    within a graph execution pipeline (QueueRunner schedule).
    
    Original Bug: ZeroBubble and DualPipeV schedules fail with torch.compiled model.
    Adapted Logic: Verify that the input producer (range_input_producer) works
    correctly with the TensorFlow graph execution scheduler (QueueRunners).
    """
    # Parameters mimicking the pipeline setup
    limit = 10
    num_epochs = 2
    batch_size = 2

    # The API under test: range_input_producer
    # This acts as the input source for the pipeline.
    queue = tf.compat.v1.train.range_input_producer(
        limit=limit,
        num_epochs=num_epochs,
        shuffle=False,
        capacity=32,
        name="input_queue"
    )

    # Define a simple pipeline stage (Dequeue + Process)
    # This mimics the model layers in the original bug.
    x = queue.dequeue_many(batch_size)
    y = x + 1  # Simple operation to simulate processing

    with tf.compat.v1.Session() as sess:
        # Initialize variables (required for num_epochs counter)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start the pipeline schedule (QueueRunners)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        results = []
        try:
            # Run the pipeline
            while True:
                try:
                    val = sess.run(y)
                    results.extend(val)
                except tf.errors.OutOfRangeError:
                    # Expected when num_epochs is exhausted
                    break
        finally:
            coord.request_stop()
            coord.join(threads)

        # Verify results
        # Expected: [0, 1, ..., 9] twice, incremented by 1 -> [1, 2, ..., 10] twice
        expected = list(range(1, limit + 1)) * num_epochs
        assert results == expected, f"Pipeline mismatch. Expected {expected}, got {results}"

if __name__ == "__main__":
    test_range_input_producer_with_pipeline_schedule()
    print("Test passed successfully.")