import tensorflow as tf
import numpy as np

def main():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    
    The original bug (Issue 161116) involves a segmentation fault when initializing
    a process group with a large number of GPUs (>40). To test the similar TensorFlow API,
    we simulate a "large scale" scenario by providing a large number of input strings
    and verifying that the queue initialization and execution do not crash.
    """
    
    # Ensure TF1 behavior is active for this legacy API
    tf.compat.v1.disable_eager_execution()

    # Simulate the "large scale" aspect of the bug (e.g., > 40 items representing GPUs)
    # The original bug triggers when scaling up resources.
    num_items = 50 
    string_tensor = tf.constant([f"file_{i}.txt" for i in range(num_items)], dtype=tf.string)

    # Initialize the string input producer
    # This corresponds to the setup phase of dist.init_process_group
    queue = tf.compat.v1.train.string_input_producer(
        string_tensor,
        num_epochs=1,
        shuffle=False,
        capacity=num_items + 10
    )

    # Define an operation to dequeue data (equivalent to engaging the communication)
    dequeue_op = queue.dequeue()

    # Initialize local variables (required for num_epochs)
    init_op = tf.compat.v1.local_variables_initializer()
    global_init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run([global_init_op, init_op])

        # Start the queue runners
        # This is the critical phase where resources are allocated, similar to the NCCL init
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            # Perform a barrier-like operation by retrieving data
            # If there is a segfault or resource allocation error, it will occur here
            result = sess.run(dequeue_op)
            
            # Assertion to verify successful operation
            assert isinstance(result, bytes), "Expected a byte string result"
            print(f"Test passed. Successfully dequeued: {result.decode('utf-8')}")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            # Clean up resources
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    main()