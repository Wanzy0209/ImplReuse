import tensorflow as tf
import numpy as np

# Disable eager execution to use compat.v1 queue-based APIs
tf.compat.v1.disable_eager_execution()

def test_string_input_producer():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    The original PyTorch issue involved complex tensor manipulations (cat, unsqueeze, arange)
    causing a compiler crash. Since string_input_producer is a data input pipeline utility
    rather than a tensor manipulation compiler, we test the robustness of the pipeline
    setup and execution, mirroring the structure of the original test (setup, execution, verification).
    """
    
    # Setup: Create a list of filenames (mimicking the data setup)
    # Using 32 items to match the dimension in the original PyTorch reproducer
    filenames = [f"file_{i}.txt" for i in range(32)]
    
    with tf.compat.v1.Session() as sess:
        # API Under Test: tf.compat.v1.train.string_input_producer
        # This creates a FIFO queue to hold the input strings
        queue = tf.compat.v1.train.string_input_producer(
            filenames, 
            num_epochs=1, 
            shuffle=False, 
            capacity=32
        )

        # Dequeue the next file name (similar to extracting results in the original)
        result = queue.dequeue()

        # Initialization: Required for local variables (epochs counter)
        sess.run([tf.compat.v1.local_variables_initializer()])
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start the queue runners to populate the queue
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        outputs = []
        try:
            # Execution: Run the pipeline to retrieve all items
            # This corresponds to the "This succeeds/crashes" execution blocks in the original
            for _ in range(32):
                val = sess.run(result)
                outputs.append(val.decode('utf-8'))
        finally:
            # Cleanup: Stop the threads
            coord.request_stop()
            coord.join(threads)

        # Verification: Ensure the pipeline produced the expected output
        assert outputs == filenames, f"Expected {filenames}, but got {outputs}"
        print("Test Passed: string_input_producer executed successfully.")

if __name__ == "__main__":
    test_string_input_producer()