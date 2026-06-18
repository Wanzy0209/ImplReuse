import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1 APIs (Sessions, Queues)
tf.compat.v1.disable_eager_execution()

def test_string_input_producer():
    # Define input data (Strings)
    # Analogous to defining tensors x, y, z in the original PyTorch test
    filenames = ["file1.txt", "file2.txt", "file3.txt"]
    string_tensor = tf.constant(filenames)

    print("Setting up input pipeline...")

    # Use the target API: tf.compat.v1.train.string_input_producer
    # This creates a queue to output the strings.
    # Analogous to torch.compile wrapping the function for optimized execution.
    input_queue = tf.compat.v1.train.string_input_producer(
        string_tensor, 
        num_epochs=1, 
        shuffle=False,
        capacity=32
    )

    # Dequeue an element to verify the pipeline works
    # Analogous to calling the compiled function
    result = input_queue.dequeue()

    # Initialize local variables (required for num_epochs counter)
    init_local = tf.compat.v1.local_variables_initializer()
    init_global = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run([init_global, init_local])
        
        # Start queue runners (essential for string_input_producer to work)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            print("Running pipeline...")
            # Run the operation to get results
            outputs = []
            for _ in range(len(filenames)):
                val = sess.run(result)
                outputs.append(val.decode())
            
            # Verify behavior
            assert outputs == filenames, f"Expected {filenames}, got {outputs}"
            print("Test passed: string_input_producer executed successfully.")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            # Stop the queue threads
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_string_input_producer()