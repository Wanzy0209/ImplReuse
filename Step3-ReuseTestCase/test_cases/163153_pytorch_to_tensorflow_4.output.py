import torch
import tensorflow as tf
import numpy as np

def verify_min_data_count(data_list, min_count=2):
    """Verification that we have at least 2 data points to run the example."""
    return len(data_list) >= min_count

def configure_input_producer(string_tensor, num_epochs=None, shuffle=True, capacity=32):
    """
    Configures the string input producer (prefetch mechanism).
    Analogous to setting up prefetch modules in FSDP.
    """
    return tf.compat.v1.train.string_input_producer(
        string_tensor,
        num_epochs=num_epochs,
        shuffle=shuffle,
        capacity=capacity,
        name="input_producer"
    )

def main():
    # Setup dummy data (filenames)
    filenames = [f"file_{i}.txt" for i in range(10)]
    _min_data_count = 2

    if not verify_min_data_count(filenames, min_count=_min_data_count):
        print(f"Unable to locate sufficient {_min_data_count} data points. Exiting.")
        return

    # Initialize the graph (analogous to torch.distributed.init_process_group)
    # Note: In TF v1, we don't explicitly init a process group like PyTorch,
    # but we start a Session and QueueRunners.
    with tf.compat.v1.Session() as sess:
        # Initialize local variables (needed for num_epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Configure the producer (The "API Under Test")
        # We set a capacity to simulate prefetching behavior
        queue = configure_input_producer(filenames, num_epochs=1, shuffle=False, capacity=5)

        # Start the queue runners (analogous to starting the distributed training)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # Verify behavior (analogous to inspect_model/running forward pass)
        results = []
        try:
            # Dequeue elements to verify the prefetch/queue mechanism works
            for _ in range(len(filenames)):
                filename_tensor = queue.dequeue()
                filename = sess.run(filename_tensor)
                results.append(filename.decode('utf-8'))
        except tf.errors.OutOfRangeError:
            print("End of queue reached.")
        finally:
            coord.request_stop()
            coord.join(threads)

        # Assertions to verify the API works correctly
        assert len(results) == len(filenames), f"Expected {len(filenames)} results, got {len(results)}"
        assert results == filenames, "Data mismatch: shuffle was False but order changed."
        print("Test passed: string_input_producer successfully prefetched and dequeued data.")

if __name__ == "__main__":
    main()