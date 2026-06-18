import tensorflow as tf
import numpy as np

def test_range_input_producer():
    """
    Adapted test case for tf.compat.v1.train.range_input_producer.
    
    The original PyTorch bug report (Issue 163687) involves specific tensor shapes 
    (e.g., batch_size=27, sequence_lengths=122, 124, 248) and an OOM error during 
    compilation. 
    
    This test adapts the dimensional constraints to the TensorFlow API. 
    We use the maximum dimension found in the original bug (248) as the limit 
    for the producer and the batch size (27) for the dequeue operation to 
    verify the API's stability with similar data scales.
    """
    
    # Disable eager execution to use tf.compat.v1 queues effectively
    # This mimics the "compiled" graph mode where the original PyTorch bug occurred.
    tf.compat.v1.disable_eager_execution()

    # Parameters derived from the PyTorch bug report
    # Original shapes: (27, 26, 62, 122), (27, 26, 124, 122), (27, 26, 248, 122)
    batch_size = 27
    limit = 248  # Using the largest dimension from the bug report
    capacity = 32 + 3 * batch_size
    num_epochs = 1

    with tf.compat.v1.Session() as sess:
        # Create the range input producer
        # This API generates integers from 0 to limit-1.
        producer = tf.compat.v1.train.range_input_producer(
            limit=limit,
            num_epochs=num_epochs,
            shuffle=False, # Keep deterministic for testing
            capacity=capacity,
            name="flex_attention_sim_producer"
        )

        # Dequeue a batch, mimicking the consumption of tensors in the original test
        # The original test performed multiple operations; here we perform a dequeue.
        batch = producer.dequeue_many(batch_size)

        # Initialize local variables (required for num_epochs)
        sess.run([tf.compat.v1.local_variables_initializer(),
                  tf.compat.v1.global_variables_initializer()])

        # Start queue runners
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            # Execute the graph (analogous to running the compiled function)
            # We run it multiple times to simulate the multiple flex_attention calls in the original bug.
            for _ in range(5):
                result = sess.run(batch)
                
                # Verify output shape matches the batch size from the bug report
                assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
                
                # Verify values are within the expected limit
                assert np.all(result >= 0) and np.all(result < limit), "Values out of range"
                
            print("Test passed: API handled batch size and limits derived from the bug report.")

        except tf.errors.OutOfRangeError:
            print("Epoch finished successfully.")
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_range_input_producer()