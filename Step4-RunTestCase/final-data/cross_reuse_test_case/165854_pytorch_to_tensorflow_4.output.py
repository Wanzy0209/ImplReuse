import tensorflow as tf
import numpy as np

# Disable eager execution to use TF1 APIs and graph mode
tf.compat.v1.disable_eager_execution()


def run_with_string_list(string_list, sess):
    """
    Run string_input_producer with a specific list of strings.
    This mimics the 'run_with_head_count' function where the buffer size (head_scale)
    changes based on the dynamic parameter H.
    """
    # Create a constant tensor instead of a placeholder.
    # Placeholders require feed_dict which is not accessible by the background
    # queue runner threads. Using a constant embeds the data in the graph.
    string_tensor = tf.constant(string_list, dtype=tf.string, name="input_strings")

    # The API under test: string_input_producer
    # We use shuffle=False to make verification deterministic
    queue = tf.compat.v1.train.string_input_producer(
        string_tensor,
        num_epochs=1,
        shuffle=False,
        capacity=32
    )

    # Dequeue operation to retrieve the strings
    dequeue_op = queue.dequeue()

    # Initialize local variables (required for num_epochs counter)
    init_local = tf.compat.v1.local_variables_initializer()
    init_global = tf.compat.v1.global_variables_initializer()

    # Initialize variables for this run
    sess.run([init_local, init_global])

    # Start queue runners
    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    print(f"  Running with list size={len(string_list)}")

    try:
        # Run iterations to consume the queue
        # In the PyTorch code, they run 5 iterations. Here we run until the queue is empty
        # (which is determined by the size of the input list).
        for i in range(len(string_list)):
            # No feed_dict needed here since we used tf.constant
            result = sess.run(dequeue_op)
            expected = string_list[i]
            # Verify the output matches the input
            assert result.decode('utf-8') == expected, f"Expected {expected}, got {result}"

        print(f"   Completed {len(string_list)} iterations")

    except tf.errors.OutOfRangeError:
        # Expected when num_epochs is exhausted
        print(f"   Reached end of queue")
    finally:
        coord.request_stop()
        coord.join(threads)


def main():
    # Test with different list counts - this makes the input size dynamic
    # and the captured buffer (queue) changes size with the list
    string_lists = [
        ["file_0", "file_1", "file_2", "file_3"],       # Size 4
        ["file_0", "file_1", "file_2", "file_3", 
         "file_4", "file_5", "file_6", "file_7"],      # Size 8
        ["file_0", "file_1", "file_2", "file_3"],       # Size 4
        ["file_" + str(i) for i in range(16)],          # Size 16
        ["file_0", "file_1", "file_2", "file_3"]        # Size 4
    ]

    print(f"Running string_input_producer with dynamic list sizes")
    print(f"Testing list sizes: {[len(l) for l in string_lists]}\n")

    for iteration, current_list in enumerate(string_lists, start=1):
        print(f"Iteration {iteration}:")
        # Create a new session for each run to ensure clean state for the queue,
        # mimicking the isolation of the function calls in the original PyTorch test.
        with tf.compat.v1.Session() as sess:
            run_with_string_list(current_list, sess)


if __name__ == "__main__":
    main()